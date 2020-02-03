#!/usr/bin/env python3
import argparse
import sys

from pixelmodels.train_common import *


def main(_=[]):
    # argument parsing
    parser = argparse.ArgumentParser(description='train nofu: a no-reference video quality model',
                                     epilog="stg7 2019",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    a = vars(parser.parse_args())
