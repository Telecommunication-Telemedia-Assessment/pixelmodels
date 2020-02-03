#!/usr/bin/env python3

from quat.ff.convert import (
    crop_video,
    convert_to_avpvs,
    convert_to_avpvs_and_crop
)
from quat.video import *
from quat.utils.assertions import *


def extract_features_no_ref(video, tmpfolder="./tmp", features_temp_folder="./tmp/features", features=None, modelname="nofu"):
    msg_assert(features is not None, "features are required to be defined", f"feature set ok")
    lInfo(f"handle : {video} for {modelname}")
    features_to_calculate = set([f for f in features.keys() if not features[f].load(features_temp_folder + "/" + modelname + "/" + f, video, f)])
    i = 0

    lInfo(f"calculate missing features {features_to_calculate} for {video}")
    if features_to_calculate != set():
        # convert to avpvs (rescale) and crop
        video_avpvs_crop = convert_to_avpvs_and_crop(video, tmpfolder + "/crop/")

        for frame in iterate_by_frame(video_avpvs_crop, convert=False):
            for f in features_to_calculate:
                x = features[f].calc(frame)
                lInfo(f"handle frame {i} of {video}: {f} -> {x}")
            i += 1
        os.remove(video_avpvs_crop)

    feature_files = []
    for f in features:
        feature_files.append(features[f].store(features_temp_folder + "/" + modelname + "/" + f, video, f))

    pooled_features = {}
    per_frame_features = {}
    for f in features:
        pooled_features = dict(advanced_pooling(features[f].get_values(), name=f), **pooled_features)
        per_frame_features = dict({f:features[f].get_values()}, **per_frame_features)

    full_features = {
        "video_name": video,
        "per_frame": per_frame_features
    }
    return pooled_features, full_features