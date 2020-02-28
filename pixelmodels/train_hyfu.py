#!/usr/bin/env python3
import argparse
import sys
import multiprocessing
from multiprocessing import Pool

import pandas as pd

from quat.log import *
from quat.parallel import run_parallel

from pixelmodels.common import get_repo_version
from pixelmodels.train_common import *
from pixelmodels.hyfu import (
    hyfu_features,
    HYFU_MODEL_PATH
)


def main(_=[]):
    # argument parsing
    parser = argparse.ArgumentParser(description='train hyfu: a hybrid no-reference video quality model',
                                     epilog=f"stg7 2020 {get_repo_version()}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("database", type=str, help="training database csv file (consists of video segment and rating value)")
    parser.add_argument("--feature_folder", type=str, default="features/train_hyfu", help="folder for storing the features")
    parser.add_argument("--temp_folder", type=str, default="tmp/train_hyfu", help="temp folder")
    parser.add_argument("--train_repetitions", type=int, default=1, help="number of repeatitions for training")
    parser.add_argument("--model", type=str, default=HYFU_MODEL_PATH, help="output model folder")
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count() // 2, help='thread/cpu count')

    a = vars(parser.parse_args())

    # read videos with training targets
    train_videos = read_train_database(a["database"])
    lInfo(f"train on {len(train_videos)} videos")

    run_parallel(
        items=train_videos,
        function=calc_and_store_features,
        arguments=[a["feature_folder"], a["temp_folder"], hyfu_features(), "hyfu", True],
        num_cpus=a["cpu_count"]
    )

    # read all features from feature folder
    features = load_features(a["feature_folder"])
    lInfo(f"loaded {len(features)} feature values")

    train_rf_models(
        features,
        num_trees=60,
        threshold="0.0001*mean",
        modelfolder=a["model"]
    )




