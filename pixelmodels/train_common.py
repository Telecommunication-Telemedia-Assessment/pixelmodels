#!/usr/bin/env python3
import datetime

import pandas as pd

from quat.log import *
from quat.utils.assertions import *
from quat.utils.system import lglob
from quat.unsorted import jdump_file

from pixelmodels.common import (
    extract_features_no_ref,
    get_repo_version
)


def calc_and_store_features_no_ref(video_and_rating, feature_folder, temp_folder, features=None, modelname="nofu"):
    msg_assert(features is not None, "features need to be defined", "features ok")
    json_assert(video_and_rating, ["video", "mos", "rating_dist", "mos_class"])

    video = video_and_rating["video"]
    assert_file(video, f"""video {video_and_rating["video"]} does not exist """, True)

    video_base_name = os.path.basename(os.path.dirname(video)) + "_" + os.path.basename(video)

    feature_filename = f"{feature_folder}/{video_base_name}.json"
    feature_filename_full = feature_filename + ".full"
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

    jdump_file(full_report_filename, full_features)
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
    assert_file(feature_folder, f"feature folder does not exist {feature_folder}", True)
    features = []
    for features_filename in lglob(feature_folder + "/*.json"):
        with open(features_filename) as feature_file:
            jfeat = json.load(feature_file)
            features.append(jfeat)
    return features


def train_rf_model(features,
        clipping=True,
        num_trees=60,
        threshold="0.0001*mean",
        graphs=True,
        save_model=True,
        target_cols=["mos", "rating_dist", "mos_class"],
        exclude_cols=["video"],
        modelfolder="models"):

    df = pd.DataFrame(features)
    print(df.head())

    params = {
        "clipping": str(clipping),
        "targets": target_cols,
        "num_trees": num_trees,
        "threshold": threshold,
        "repo_version": get_repo_version(),
        "date": str(datetime.datetime.now()),
    }

    # train model


    # store plots

    # store results

    # store general model info
    jdump_file(modelfolder + "/info.json", params)



