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

from quat.visual.base_features import *
from quat.visual.image import *
from quat.ff.probe import *
from quat.ml.mlcore import *
from quat.video import *

from pixelmodels.common import extract_features_no_ref

NOFU_MODEL_PATH = os.path.dirname(__file__) + "/models/nofu/"
# for each type of model a subpath


def nofu_features():
    return {
        "contrast",
        "fft",
        "blur",
        "color_fulness",
        "saturation",
        "tone",
        "scene_cuts",
        "movement",
        "temporal",
        "si",
        "ti",
        "blkmotion",
        "cubrow.0",
        "cubcol.0",
        "cubrow.1.0",
        "cubcol.1.0",
        "cubrow.0.3",
        "cubcol.0.3",
        "cubrow.0.6",
        "cubcol.0.6",
        "cubrow.0.5",
        "cubcol.0.5",
        "staticness",
        "uhdhdsim",
        "blockiness",
        # TODO: noise
    }
    # removed and multi-value features
    # "niqe": ImageFeature(calc_niqe_features),
    # "brisque": ImageFeature(calc_brisque_features),
    # "ceiq": ImageFeature(ceiq)
    # "strred": StrredNoRefFeatures(),


def nofu_predict_video_score(video, tmpfolder="./tmp", features_temp_folder="./tmp/features", clipping=True):
    features, full_report = extract_features_no_ref(
        video,
        tmpfolder=tmpfolder,
        features_temp_folder=features_temp_folder,
        featurenames=nofu_features(),
        modelname="nofu"
    )
    # predict quality
    df = pd.DataFrame([features])
    columns = df.columns.difference(["rating", "filename"])
    X = df[sorted(columns)]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values

    # TODO: fix
    model = load_serialized(get_latest_nofu_model())

    predicted = model.predict(X)
    # apply clipping if needed
    if clipping:
        predicted = np.clip(predicted, 1, 5)

    return predicted



def main(_=[]):
    # argument parsing
    parser = argparse.ArgumentParser(description='nofu: a no-reference video quality model',
                                     epilog="stg7 2019",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", type=str, nargs="+", help="video to predict video quality")
    parser.add_argument("--feature_filename", type=str, default=None, help="store features in a file, e.g. for training an own model")
    parser.add_argument("--temp_folder", type=str, default="./tmp", help="temp folder for intermediate results")
    parser.add_argument("--model", type=str, default=NOFU_MODEL_PATH, help="specified pre-trained model")
    parser.add_argument('--output_report', type=str, default="report.json", help="output report of calculated values")
    parser.add_argument('--clipping', action='store_true', help="use clipping of final scores, to [1,5]")
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count() // 2, help='thread/cpu count')

    a = vars(parser.parse_args())

    for video in a["video"]:
        extract_features_no_ref(
            video,
            features=nofu_features()
        )




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
