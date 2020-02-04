#!/usr/bin/env python3
import pandas as pd

from quat.log import *
from quat.utils.assertions import *
from quat.utils.system import lglob
from quat.unsorted import jdump_file

from pixelmodels.common import extract_features_no_ref


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
    #msg_assert("ratings" in df.columns, "ratings needs to be part of database file", "video_name ok")
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
            "rating_dist": [],
        }
        )
    return videos