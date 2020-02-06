#!/usr/bin/env python3
import datetime

import pandas as pd

from quat.log import *
from quat.utils.assertions import *
from quat.utils.system import lglob
from quat.unsorted import jdump_file
from quat.ml.mlcore import (
    train_rf_class,
    train_rf_regression,
    eval_plots_class,
    eval_plots_regression,
    save_serialized
)

from pixelmodels.common import (
    extract_features_no_ref,
    get_repo_version
)


def calc_and_store_features_no_ref(video_and_rating, feature_folder, temp_folder, features=None, modelname="nofu"):
    msg_assert(features is not None, "features need to be defined", "features ok")
    json_assert(video_and_rating, ["video", "mos", "rating_dist", "mos_class"])

    video = video_and_rating["video"]
    assert_file(video, True)

    video_base_name = os.path.basename(os.path.dirname(video)) + "_" + os.path.basename(video)

    pooled_features_filename = f"{feature_folder}/{video_base_name}.json"
    full_features_filename = pooled_features_filename + ".full"

    if os.path.isfile(pooled_features_filename) and os.path.isfile(full_features_filename):
        lInfo(f"features are already calculated, so use cached values, if this is not needed please delete {pooled_features_filename}")
        with open(pooled_features_filename) as pfp:
            pooled_features = json.load(pfp)
        return pooled_features

    pooled_features, full_features = extract_features_no_ref(
        video,
        temp_folder,
        feature_folder,
        features,
        modelname
    )

    if pooled_features is None or full_features is None:
        lWarn(f"features and full_feature are empty, something wrong for {video}")
        return None

    pooled_features["mos"] = video_and_rating["mos"]
    pooled_features["rating_dist"] = video_and_rating["rating_dist"]
    pooled_features["mos_class"] = video_and_rating["mos_class"]

    jdump_file(full_features_filename, full_features)
    jdump_file(pooled_features_filename, pooled_features)

    return pooled_features


def read_train_database_no_ref(database):
    df = pd.read_csv(database)
    msg_assert("MOS" in df.columns or "mos" in df.columns, "MOS needs to be part of database file", "MOS ok")
    msg_assert("video_name" in df.columns, "video_name needs to be part of database file", "video_name ok")

    mos_col = "MOS" if "MOS" in df.columns else "mos"
    # individual ratings
    user_cols = [x for x in df.columns if "user" in x]
    if len(user_cols) == 0:
        lWarn("rating distribution cannot be used for training, they are not part of the given database file")

    videos = []
    for _, i in df.iterrows():
        video_filename_path = os.path.join(
            os.path.dirname(database),
            "segments",
            i["video_name"]
        )
        rating_dist = {}
        for ucol in user_cols:
            rating_dist[i[ucol]] = rating_dist.get(i[ucol], 0) + 1
        assert_file(video_filename_path, True)

        videos.append({
            "video": video_filename_path,
            "mos": i[mos_col],
            "mos_class": int(round(i[mos_col], 0)),
            "rating_dist": rating_dist,
        })
    return videos


def load_features(feature_folder):
    assert_dir(feature_folder, True)
    features = []
    for features_filename in lglob(feature_folder + "/*.json"):
        with open(features_filename) as feature_file:
            jfeat = json.load(feature_file)
            features.append(jfeat)
    return features


def train_rf_models(features,
        clipping=True,
        num_trees=60,
        threshold="0.0001*mean",
        graphs=True,
        save_model=True,
        target_cols=["mos", "rating_dist", "mos_class"],
        exclude_cols=["video"],
        modelfolder="models"):

    os.makedirs(modelfolder, exist_ok=True)

    df = pd.DataFrame(features)
    df.to_csv(os.path.join(modelfolder, "features.csv"), index=False)

    print(df.head())
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
        Y = df[model]  # target col as model name

        if "_class" in model:
            lInfo(f"train {model} as classification")
            result = train_rf_class(X, Y, num_trees, threshold)
            save_serialized(result["randomforest"], models["_class"])
            cval = result["crossval"]
            metrics = eval_plots_class(cval["truth"], cval["predicted"], title=model, folder=modelfolder + "/_class/")
            cval.to_csv(modelfolder + "/crossval_class.csv", index=False)
            params["class_performance"] = metrics
            continue
        if "_dist" in model:
            lInfo(f"train {model} as multi instance regression")

            continue
        # default case: regression
        lInfo(f"train {model} as regression")
        result = train_rf_regression(X, Y, num_trees, threshold)
        save_serialized(result["randomforest"], models["regression"])
        cval = result["crossval"]
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



