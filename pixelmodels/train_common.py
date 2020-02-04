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
    json_assert(video_and_rating, ["video", "mos", "rating_dist"])

    video = video_and_rating["video"]
    assert_file(video, f"""video {video_and_rating["video"]} does not exist """)

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

    jdump_file(full_report_filename, full_features)
    jdump_file(pooled_features_filename, pooled_features)

    return pooled_features


def read_train_database_no_ref(database):
    df = pd.read_csv(database)
    msg_assert("MOS" in df.columns, "MOS needs to be part of database file", "MOS ok")
    msg_assert("video_name" in df.columns, "video_name needs to be part of database file", "video_name ok")

    # individual ratings
    rating_dist = []
    if "ratings" in df.columns:
        # TODO
        rating_dist = []
    else:
        lWarn("rating distribution cannot be used for training, they are not part of the given database file")
    videos = []
    for _, i in df[["video_name", "MOS"]].iterrows():
        video_filename_path = os.path.join(
            os.path.dirname(database),
            "segments",
            i["video_name"]
        )
        assert_file(video_filename_path)

        videos.append({
            "video": video_filename_path,
            "mos": i["MOS"],
            "rating_dist": rating_dist,
        })
    return videos


def load_features(feature_folder):
    assert_file(feature_folder, f"feature folder does not exist {feature_folder}")
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
        target_cols=["mos", "rating_dist"],
        exclude_cols=["video"],
        modelfolder="models"):

    df = pd.DataFrame(features)
    print(df.head())

    params = {
        "clipping": str(clipping),
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



