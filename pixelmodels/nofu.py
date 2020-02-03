#!/usr/bin/env python3
# nofu -- no-reference video quality model
import argparse
import sys
import json
import datetime
import os
import traceback
import shutil
import multiprocessing

import numpy as np
import pandas as pd


from quat.log import *
from quat.utils.system import *
from quat.unsorted import *
from quat.parallel import *
from quat.video import *
from quat.ff.probe import *
from quat.ml.mlcore import *

from quat.ff.convert import (
    crop_video,
    convert_to_avpvs,
    convert_to_avpvs_and_crop
)

from quat.visual.base_features import *
from quat.visual.image import *

def nofu_features():
    return {
        "contrast": ImageFeature(calc_contrast_features),
        "fft": ImageFeature(calc_fft_features),
        "blur": ImageFeature(calc_blur_features),
        "color_fulness": ImageFeature(color_fulness_features),
        "saturation": ImageFeature(calc_saturation_features),
        "tone": ImageFeature(calc_tone_features),
        "scene_cuts": CutDetectionFeatures(),
        "movement": MovementFeatures(),
        "temporal": TemporalFeatures(),
        "si": SiFeatures(),
        "ti": TiFeatures(),
        "blkmotion": BlockMotion(),
        "cubrow.0": CuboidRow(0),
        "cubcol.0": CuboidCol(0),
        "cubrow.1.0": CuboidRow(1.0),
        "cubcol.1.0": CuboidCol(1.0),
        "cubrow.0.3": CuboidRow(0.3),
        "cubcol.0.3": CuboidCol(0.3),
        "cubrow.0.6": CuboidRow(0.6),
        "cubcol.0.6": CuboidCol(0.6),
        "cubrow.0.5": CuboidRow(0.5),
        "cubcol.0.5": CuboidCol(0.5),
        "staticness": Staticness(),
        "uhdhdsim": UHDSIM2HD(),
        "blockiness": Blockiness(),
    }
    # removed and multi-value features
    # "niqe": ImageFeature(calc_niqe_features),
    # "brisque": ImageFeature(calc_brisque_features),
    # "ceiq": ImageFeature(ceiq)
    # "strred": StrredNoRefFeatures(),



def extract_features_no_ref(video, tmpfolder="./tmp", features_temp_folder="./tmp/features", features=nofu_features()):
    lInfo(f"handle : {video}")
    features_to_calculate = set([f for f in features.keys() if not features[f].load(features_temp_folder + "/" + f, video, f)])
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
        feature_files.append(features[f].store(features_temp_folder + "/" + f, video, f))

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


def nofu_predict_video_score(video, model, tmpfolder="./tmp", features_temp_folder="./tmp/features", clipping=True):
    features, full_report = extract_features(
        video,
        tmpfolder=tmpfolder,
        features_temp_folder=features_temp_folder,
        features=nofu_features()
    )
    # predict quality
    df = pd.DataFrame([features])
    columns = df.columns.difference(["rating", "filename"])
    X = df[sorted(columns)]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
    predicted = model.predict(X)
    # apply clipping if needed
    if clipping:
        predicted = np.clip(predicted, 1, 5)

    per_frame_df = pd.DataFrame({
        "si" : full_report["per_frame"]["si"],
        "ti": full_report["per_frame"]["ti"]
    })
    per_frame_df["predicted"] = predicted[0]
    #per_frame_df.to_csv("persecondthing.csv", index=False)

    from scipy.signal import savgol_filter

    # for ti the relationship is inverse
    per_frame_df["ti"] = per_frame_df["ti"].max() - per_frame_df["ti"]
    # window size 21, polynomial order 3
    window_size = min(51,  2 * (len(per_frame_df["ti"]) // 2) - 1)
    lInfo(f"window size: {window_size}")

    if window_size <= 3:
        per_frame_df["si_smooth"] = per_frame_df["si"]
        per_frame_df["ti_smooth"] = per_frame_df["ti"]
    else:
        per_frame_df["si_smooth"] = savgol_filter(per_frame_df["si"], window_size, 3)
        per_frame_df["ti_smooth"] = savgol_filter(per_frame_df["ti"], window_size, 3)


    def mos_scaling(values, predicted):
        mean_v = np.array(values).mean()
        v = np.clip(predicted * values / mean_v, 1, 5)
        v[np.isnan(v)] = predicted
        return v

    si_mos = mos_scaling(per_frame_df["si_smooth"], predicted[0])
    ti_mos = mos_scaling(per_frame_df["ti_smooth"], predicted[0])

    per_frame_df["per_frame_mos"] =  2 / 3 * si_mos + 1 / 3 * ti_mos

    return predicted, per_frame_df["per_frame_mos"].values, features, full_report


def main(_=[]):
    # argument parsing
    parser = argparse.ArgumentParser(description='nofu: a no-reference video quality model',
                                     epilog="stg7 2019",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs="+", help="video to predict video quality")
    parser.add_argument("--feature_filename", type=str, default=None, help="store features in a file, e.g. for training an own model")
    parser.add_argument("--temp_folder", type=str, default="./tmp", help="temp folder for intermediate results")
    parser.add_argument("--model", type=str, default="models/nofu/model.npz", help="specified pre-trained model")
    parser.add_argument('--output_report', type=str, default="report.json", help="output report of calculated values")
    parser.add_argument('--clipping', action='store_true', help="use clipping of final scores, to [1,5]")
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count() // 2, help='thread/cpu count')

    a = vars(parser.parse_args())
    run_parallel(
        a["video"],
        extract_features_no_ref,
        num_cpus=a["cpu_count"]
    )



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
