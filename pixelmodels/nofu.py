#!/usr/bin/env python3
# nofu -- no-reference video quality model
import argparse
import sys
import os
import multiprocessing

from quat.log import *
from quat.utils.system import *
from quat.utils.fileutils import *
from quat.unsorted import *
from quat.parallel import *
from quat.ml.mlcore import *
from quat.unsorted import jdump_file

from pixelmodels.train_common import (
    read_database
)
from pixelmodels.common import (
    extract_features_no_ref,
    get_repo_version,
    predict_video_score,
    MODEL_BASE_PATH
)

# this is the basepath, so for each type of model a separate file is stored
NOFU_MODEL_PATH = os.path.join(MODEL_BASE_PATH, "nofu")


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
        "noise",
        "brisque",
        #"ceiq", # no improvement
        #"compressibility", # no improvement
        # "niqe", # brisque is enough
        # not sure about the following features
        #"strred"
    }



def nofu_predict_video_score(video, temp_folder="./tmp", features_temp_folder="./tmp/features", model_path=NOFU_MODEL_PATH, clipping=True):
    features, full_report = extract_features_no_ref(
        video,
        temp_folder=temp_folder,
        features_temp_folder=features_temp_folder,
        featurenames=nofu_features(),
        modelname="nofu"
    )
    return predict_video_score(features, model_path)


def main(_=[]):
    # argument parsing
    parser = argparse.ArgumentParser(
        description='nofu: a no-reference video quality model',
        epilog=f"stg7 2020 {get_repo_version()}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--feature_folder", type=str, default="./features/nofu", help="store features in a file, e.g. for training an own model")
    parser.add_argument("--temp_folder", type=str, default="./tmp/nofu", help="temp folder for intermediate results")
    parser.add_argument("--model", type=str, default=NOFU_MODEL_PATH, help="specified pre-trained model")

    subparsers = parser.add_subparsers(
        help='sub commands',
        dest="command"
    )

    predict = subparsers.add_parser(
        'predict',
        help='predict video quality of single video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    predict.add_argument(
        'video',
        type=str,
        help='video to predict video quality'
    )
    predict.add_argument(
        '--output_report',
        type=str,
        default=None,
        help="output report of calculated values, None uses the video name as basis"
    )

    batch = subparsers.add_parser(
        'batch',
        help='perform batch prediction of a full database',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    batch.add_argument(
        'database',
        type=str,
        help='csv file of database, e.g. per_user.csv'
    )
    batch.add_argument(
        '--cpu_count',
        type=int,
        default=multiprocessing.cpu_count() // 2,
        help='thread/cpu count'
    )
    batch.add_argument(
        '--output_report_folder',
        type=str,
        default="reports/nofu",
        help="folder for output reports of calculated values, video name is used as basis"
    )

    a = vars(parser.parse_args())

    if a["command"] == "predict":
        if a["output_report"] is None:
            a["output_report"] = get_filename_without_extension(a["video"]) + ".json"

        prediction = nofu_predict_video_score(
            a["video"],
            temp_folder=a["temp_folder"],
            features_temp_folder=a["feature_folder"],
            model_path=a["model"],
            clipping=True
        )
        jprint(prediction)
        jdump_file(a["output_report"], prediction)

    if a["command"] == "batch":
        lInfo("batch prediction")
        videos = [x["video"] for x in read_database(a["database"])]
        results = run_parallel(
            items=videos,
            function=nofu_predict_video_score,
            arguments=[a["temp_folder"], a["feature_folder"], a["model"], True],
            num_cpus=a["cpu_count"]
        )
        os.makedirs(a["output_report_folder"], exist_ok=True)
        for i, result in enumerate(results):
            dn = os.path.normpath(os.path.dirname(videos[i])).replace(os.sep, "_")
            report_filename = dn + get_filename_without_extension(videos[i]) + ".json"
            jdump_file(
                os.path.join(a["output_report_folder"], report_filename),
                result
            )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
