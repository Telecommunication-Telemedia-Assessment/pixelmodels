#!/usr/bin/env python3
import datetime

import pandas as pd

from quat.log import *
from quat.utils.assertions import *
from quat.utils.fileutils import *
from quat.utils.system import lglob
from quat.unsorted import jdump_file
from quat.ml.mlcore import (
    train_rf_class,
    train_rf_regression,
    train_rf_multi_regression,
    eval_plots_class,
    eval_plots_regression,
    save_serialized
)

from pixelmodels.common import (
    extract_features_no_ref,
    extract_features_full_ref,
    get_repo_version
)



def calc_and_store_features(video_and_rating, feature_folder, temp_folder, features=None, modelname="nofu", meta=False):
    """
    calcualtes and stores features of the given video, in case features are already stored, reuse the stored ones

    video_and_rating is a dictionary containing:
        video_and_rating["video"]: video_filename_path
        video_and_rating["mos"]: mos score
        video_and_rating["rating_dist"]: rating_dist values
        video_and_rating["mos_class"]: classification score

        in case of a full-reference video quality model:
        video_and_rating["src_video"]: source video
    """
    msg_assert(features is not None, "features need to be defined", "features ok")
    json_assert(video_and_rating, ["video", "mos", "rating_dist", "mos_class"])

    full_ref = "src_video" in video_and_rating
    video = video_and_rating["video"]
    assert_file(video, True)

    dn = os.path.normpath(os.path.dirname(video)).replace(os.sep, "_")
    video_base_name = dn + "_" + os.path.basename(os.path.splitext(video)[0])

    pooled_features_filename = f"{feature_folder}/{video_base_name}.json"
    full_features_filename = pooled_features_filename + ".full"

    if os.path.isfile(pooled_features_filename) and os.path.isfile(full_features_filename):
        lInfo(f"features are already calculated, so use cached values, if this is not needed please delete {pooled_features_filename}")
        with open(pooled_features_filename) as pfp:
            pooled_features = json.load(pfp)
        return pooled_features
    if full_ref:
        pooled_features, full_features = extract_features_full_ref(
            video,
            video_and_rating["src_video"],
            temp_folder,
            feature_folder,
            features,
            modelname,
            meta
        )
    else:
        pooled_features, full_features = extract_features_no_ref(
            video,
            temp_folder,
            feature_folder,
            features,
            modelname,
            meta
        )

    if pooled_features is None or full_features is None:
        lWarn(f"features or full_feature are empty, something wrong for {video}")
        return None

    if full_ref:
        pooled_features["src_video"] = video_and_rating["src_video"]

    pooled_features["mos"] = video_and_rating["mos"]
    pooled_features["rating_dist"] = video_and_rating["rating_dist"]
    pooled_features["mos_class"] = video_and_rating["mos_class"]

    jdump_file(full_features_filename, full_features)
    jdump_file(pooled_features_filename, pooled_features)

    return pooled_features


def read_database(database, full_ref=False):
    """
    reads a databases

    in case full_ref: then also src videos are loaded

    returns list of dicts with
        video
        mos
        mos_class
        rating_dist
        src_video: optional
    """
    df = pd.read_csv(database)
    msg_assert("MOS" in df.columns or "mos" in df.columns, "MOS needs to be part of database file", "MOS ok")
    msg_assert("video_name" in df.columns, "video_name needs to be part of database file", "video_name ok")

    mos_col = "MOS" if "MOS" in df.columns else "mos"
    # individual ratings
    user_cols = [x for x in df.columns if "user" in x]
    if len(user_cols) == 0:
        lWarn("rating distribution cannot be used for training, they are not part of the given database file")

    videos = []
    dirname_database = os.path.dirname(database)
    for _, i in df.iterrows():
        video_filename_path = os.path.join(
            dirname_database,
            "segments",
            i["video_name"]
        )
        rating_dist = {}
        for ucol in user_cols:
            rating_dist[i[ucol]] = rating_dist.get(i[ucol], 0) + 1
        assert_file(video_filename_path, True)
        video = {
            "video": video_filename_path,
            "mos": i[mos_col],  # will be handled as regression
            "mos_class": int(round(i[mos_col], 0)),  # will be handled as classicication
            "rating_dist": rating_dist,  # will be handled as multi instance regression
        }
        if full_ref:
            src_video_pattern = dirname_database + f"/../src_videos/*"
            src_videos = lglob(src_video_pattern)
            matching_src_videos = list(filter(
                lambda x: get_filename_without_extension(x) in i["video_name"],
                src_videos
                )
            )
            msg_assert(len(matching_src_videos) >= 1, f"something wrong with src video mapping; check: {i['video_name']}")
            # take longest matching src_video (in case two are matching)
            matching_src_videos = sorted(matching_src_videos)
            video["src_video"] = os.path.abspath(matching_src_videos[-1])
        videos.append(video)
    return videos


def load_features(feature_folder):
    """
    loads feature values from a folder,
    here it is assumed that pooled features are plain json files
    """
    assert_dir(feature_folder, True)
    features = []
    for features_filename in lglob(feature_folder + "/*.json"):
        with open(features_filename) as feature_file:
            jfeat = json.load(feature_file)
            features.append(jfeat)
    return features


def convert_dist(y_values):
    """
    convert and unify distribution values of individual ratings
    """
    def unify_dist(y, range_values):
        sum_ = sum([y[x] for x in y])
        for x in range_values:
            y[x] = y.get(x, 0) / sum_
        return y
    range_values = set(sum([list(y.keys()) for y in y_values], []))
    values = [unify_dist(y, range_values) for y in y_values]
    return pd.DataFrame(values)


def train_rf_models(features,
        clipping=True,
        num_trees=60,
        threshold="0.0001*mean",
        graphs=True,
        save_model=True,
        target_cols=["mos", "rating_dist", "mos_class"],
        exclude_cols=["video", "src_video"],
        modelfolder="models"):
    """
    train several random forest models (for each traget column one model,
    depending on the given input values) to predict video quality
    """
    os.makedirs(modelfolder, exist_ok=True)

    df = pd.DataFrame(features)
    df.to_csv(os.path.join(modelfolder, "features.csv"), index=False)

    lInfo(df.head())
    models_to_train = [x for x in df.columns if x in target_cols]
    if len(set(target_cols) & set(models_to_train)) != len(target_cols):
        missing = set(target_cols) - set(models_to_train)
        lWarn(f"some target columns are not stored in the feature data, thus they cannot be used for training: missing cols: {missing}")

    params = {
        "clipping": str(clipping),
        "targets": target_cols,
        "num_trees": num_trees,
        "threshold": threshold,
        "models_to_train": models_to_train,
        "repo_version": get_repo_version(),
        "date": str(datetime.datetime.now()),
    }

    models = {
        "regression": modelfolder + "/model_regression.npz",
        "_class": modelfolder + "/model_class.npz",
        "_dist": modelfolder + "/model_rating_dist.npz"
    }

    feature_cols = df.columns.difference(target_cols + exclude_cols)

    X = df[sorted(feature_cols)]

    for model in models_to_train:
        Y = df[model]  # target column is the model name

        if "_class" in model:
            lInfo(f"train {model} as classification")
            result = train_rf_class(X, Y, num_trees, threshold)
            save_serialized(result["randomforest"], models["_class"])
            cval = result["crossval"]
            if clipping:
                cval["predicted"] = cval["predicted"].clip(1, 5)
            metrics = eval_plots_class(cval["truth"], cval["predicted"], title=model, folder=modelfolder + "/_class/")
            cval.to_csv(modelfolder + "/crossval_class.csv", index=False)
            params["class_performance"] = metrics
            continue
        if "_dist" in model:
            lInfo(f"train {model} as multi instance regression")
            Y = convert_dist(Y.values)
            Y = Y[sorted(Y.columns)]
            result = train_rf_multi_regression(X, Y, num_trees, threshold)
            save_serialized(result["randomforest"], models["_dist"])
            cval = result["crossval"]

            # perform per pair (predicted_N, truth_N) eval plots
            for c in cval.columns:
                if "predicted" in c:
                    continue
                truth_col = c
                predicted_col = truth_col.replace("truth", "predicted")
                part = truth_col.replace("truth_", "")
                metrics = eval_plots_regression(
                    cval[truth_col],
                    cval[predicted_col],
                    title=model + f"_{part}",
                    folder=modelfolder + "/_rating_dist/",
                    plotname=f"rf_mi_@{num_trees}_dist_{part}"
                )
                params["regression_performance_dist_" + part] = metrics
            cval.to_csv(modelfolder + "/crossval_rating_dist.csv", index=False)
            continue
        # default case: regression
        lInfo(f"train {model} as regression")
        result = train_rf_regression(X, Y, num_trees, threshold)
        save_serialized(result["randomforest"], models["regression"])
        cval = result["crossval"]
        if clipping:
            cval["predicted"] = cval["predicted"].clip(1, 5)
        metrics = eval_plots_regression(
            cval["truth"],
            cval["predicted"],
            title=model,
            folder=modelfolder + "/_reggression/",
            plotname=f"rf_@{num_trees}"
        )
        cval.to_csv(modelfolder + "/crossval_regression.csv", index=False)
        metrics["number_features"] = result["number_features"]
        metrics["used_features"] = result["used_features"]
        params["regression_performance"] = metrics

    # store general model info
    jdump_file(modelfolder + "/info.json", params)
