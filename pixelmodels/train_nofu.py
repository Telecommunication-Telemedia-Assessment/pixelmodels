#!/usr/bin/env python3
import argparse
import sys
import multiprocessing
from multiprocessing import Pool

from quat.log import *
from quat.parallel import run_parallel

from pixelmodels.train_common import *
from pixelmodels.nofu import nofu_features


def main(_=[]):
    # argument parsing
    parser = argparse.ArgumentParser(description='train nofu: a no-reference video quality model',
                                     epilog="stg7 2019",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("database", type=str, help="training database csv file (consists of video segment and rating value)")
    parser.add_argument("--feature_folder", type=str, default="features", help="folder for storing the features")
    parser.add_argument("--temp_folder", type=str, default="tmp/train_nofu", help="temp folder")
    parser.add_argument("--train_repetitions", type=int, default=1, help="number of repeatitions for training")
    parser.add_argument("--model", type=str, default="models/nofu.npz", help="output model")
    parser.add_argument('--cpu_count', type=int, default=multiprocessing.cpu_count() // 2, help='thread/cpu count')

    a = vars(parser.parse_args())

    # read videos with training targets
    train_videos = read_train_database_no_ref(a["database"])
    lInfo(f"train on {len(train_videos)} videos")

    run_parallel(
        items=train_videos,
        function=calc_and_store_features_no_ref,
        arguments=[a["feature_folder"], a["temp_folder"], nofu_features(), "nofu"],
        num_cpus=a["cpu_count"]
    )

    # read all features from feature folder



