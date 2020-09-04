#!/usr/bin/env python3
import datetime
import os
import shutil

from quat.ff.probe import ffprobe
from quat.ff.convert import (
    crop_video,
    convert_to_avpvs,
    convert_to_avpvs_and_crop
)
from quat.ml.mlcore import load_serialized
from quat.video import *
from quat.utils.fileutils import get_filename_without_extension
from quat.utils.assertions import *
from quat.utils.system import shell_call
from quat.visual.base_features import *
from quat.visual.fullref import *
from quat.visual.image import *


MODEL_BASE_PATH = os.path.abspath(os.path.dirname(__file__) + "/models")


import tempfile

class CompressibilityFeature(Feature):
    # TODO: move to quat
    # FIX: local folder usage (here a specified temporary folder would be the better approach)
    def __init__(self):
        self._values = []
        self._writer = None
        self._writer_ref = None

    def _create_video_stream(self,):
        tf = tempfile.NamedTemporaryFile()
        os.makedirs("./compressibility", exist_ok=True)
        _video_filename = "./compressibility/" + os.path.basename(tf.name) + ".mp4"
        _writer = skvideo.io.FFmpegWriter(
            _video_filename,
            inputdict={
                "-r": "60",
            },
            outputdict={
                "-r": "60",
                "-c:v": "libx264",
                "-preset": "medium",
                "-crf": "24"
            }
        )
        return _video_filename, _writer

    def calc_ref_dis(self, dframe, rframe):
        if self._writer is None:
            self._video_filename, self._writer = self._create_video_stream()
        if self._writer_ref is None:
            self._video_filename_ref, self._writer_ref = self._create_video_stream()
        self._writer.writeFrame(dframe)
        self._writer_ref.writeFrame(rframe)
        return 0 # here only fake values are returned

    def calc(self, frame, debug=False):
        if self._writer is None:
            self._video_filename, self._writer = self._create_video_stream()
        self._writer.writeFrame(frame)
        return 0 # here only fake values are returned

    def store(self, folder, video, name=""):
        if self._writer is not None:
            self._writer.close()
            filesize = os.stat(self._video_filename).st_size  / 1024 / 1024
            self._values = [filesize]

        if self._writer_ref is not None:
            # this is the full-ref case
            self._writer_ref.close()
            filesize = os.stat(self._video_filename).st_size  / 1024 / 1024
            filesize_ref = os.stat(self._video_filename_ref).st_size  / 1024 / 1024

            self._values = [{
                "diff": filesize_ref - filesize,
                "dis": filesize,
                "ref": filesize_ref
                }
            ]

        return super().store(folder, video, name)

    def get_values(self):
        # FIX: this behaviour is not how it should be
        msg_assert(len(self._values) > 0, f"this should not happen, please call store before")
        return self._values

    def __del__(self):
        try:
            # TODO: delete temp files
            self._writer.close()
            self._writer_ref.close()
        except:
            return


def get_repo_version():
    """
    returns a unified repo version for the final reports (branch and current commit sha)
    """
    this_path = os.path.dirname(__file__)
    sha = shell_call(f"cd {this_path} && git rev-parse HEAD").strip()
    branch = shell_call(f"cd {this_path} && git rev-parse --abbrev-ref HEAD").strip()
    return branch + "@" + sha


def all_no_ref_features():
    """
    returns only all no-reference features
    """
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
        "noise": ImageFeature(calc_noise),
        "niqe": ImageFeature(calc_niqe_features),
        "brisque": ImageFeature(calc_brisque_features),
        "ceiq": ImageFeature(ceiq),
        "strred": StrredNoRefFeatures(),
        "compressibility": CompressibilityFeature()
    }


def all_features():
    """
    returns all possible features, full- and no-reference
    """
    full_ref_features = {
        "ssim": SSIM(),
        "psnr": PSNR(),
        "vifp": VIFP(),
        "fps": FramerateEstimator()
    }
    return dict(all_no_ref_features(), **full_ref_features)


def unify_video_codec(video_codec_name):
    """
    maps ffmpeg video codec name to a number,

    Returns
    h264=0, h265=1, vp9=2
    """
    if "h264" in video_codec_name:
        return 0
    if "hevc" in video_codec_name:
        return 1
    if "vp9" in video_codec_name:
        return 2
    msg_assert(False, f"video codec{video_codec_name} is not supported by this model")
    # this should never happen
    return 3


def extract_mode0_features(video):
    """
    extract mode 0 base-features, e.g. framerate, bitrate

    Returns

    Dictionary with:
    - framerate : float
    - bitrate : float in kbit/s
    - codec : int (coded as integer: h264=0, h265=1, vp9=2)
    - resolution : int (height * width)
    - bpp : float, bits per pixel
    - bitrate_log : float, log of bitrate
    - framerate_norm: float, normalized framerate (fps/60)
    - framerate_log : float, log of fps
    - resolution_log : float, log of resolution
    - resolution_norm: float, resolution normalized by UHD-1/4K resolution
    """
    # use ffprobe to extract bitstream features
    meta = ffprobe(video)
    # mode0 base data
    mode0_features = {  # numbers are important here
        "framerate": float(meta["avg_frame_rate"]),
        "bitrate": float(meta["bitrate"]) / 1024,  # kbit/s
        "bitdepth": 8 if meta["bits_per_raw_sample"] == "unknown" else int(meta["bits_per_raw_sample"]),
        "codec": unify_video_codec(meta["codec"]),
        "resolution": int(meta["height"]) * int(meta["width"]),
    }
    # mode0 extended features
    mode0_features["bpp"] = 1024 * mode0_features["bitrate"] / (mode0_features["framerate"] * mode0_features["resolution"])
    mode0_features["bitrate_log"] = np.log(mode0_features["bitrate"])
    mode0_features["framerate_norm"] = mode0_features["framerate"] / 60.0
    mode0_features["framerate_log"] = np.log(mode0_features["framerate"])
    mode0_features["resolution_log"] = np.log(mode0_features["resolution"])
    mode0_features["resolution_norm"] = mode0_features["resolution"] / (3840 * 2160)

    return mode0_features


def __filter_to_be_calculated_features(video, all_feat, featurenames, features_temp_folder):
    """
    filters from the given featurenames , the features in all_feat that still need to be calculated,
    here also a loading of already calculated feature values is performed
    """
    msg_assert(len(list(set(all_feat.keys()) & featurenames)) > 0, "feature set empty")
    msg_assert(len(list(set(featurenames - all_feat.keys()))) == 0, "feature set comtains features that are not defined")
    features = {}
    for featurename in featurenames:
        features[featurename] = all_feat[featurename]

    features_to_calculate = set([f for f in features.keys() if not features[f].load(features_temp_folder + "/" + f, video, f)])
    return features_to_calculate, features


def __store_and_pool_features(video, features, meta, features_temp_folder):
    """
    stores `features` for a given `video` in the folder `features_temp_folder`, in case meta is true,
    such features will be extended by mode0 meta-data based features
    """
    feature_files = []
    for f in features:
        feature_files.append(features[f].store(features_temp_folder + "/" + f, video, f))

    pooled_features = {}
    per_frame_features = {}
    for f in features:
        values = features[f].get_values()
        # TODO: this handling should be done by advanced_pooling...
        if len(values) == 1:
            if type(values[0]) == dict:
                flattened = {
                    f + "_" + x: values[0][x] for x in values[0]
                }
                pooled_features = dict(flattened, **pooled_features)
            else:
                pooled_features = dict({f: values[0]}, **pooled_features)
        else: # stats=True, minimal=False
            #pooled_features = dict(advanced_pooling(values, name=f, stats=False, minimal=True), **pooled_features)
            pooled_features = dict(advanced_pooling(values, name=f), **pooled_features)
        per_frame_features = dict({f:values}, **per_frame_features)

    full_features = {
        "video_name": video,
        "per_frame": per_frame_features
    }
    # this is only used if it is a hybrid model, thus extend features by mode0 features
    if meta:
        metadata_features = extract_mode0_features(video)
        full_features["meta"] = metadata_features
        for m in metadata_features:
            pooled_features["meta_" + m] = metadata_features[m]
    return pooled_features, full_features


def extract_features_no_ref(video, temp_folder="./tmp", features_temp_folder="./tmp/features", featurenames=None, modelname="nofu", meta=False):
    """
    extract no-reference features for a given video.
    use `temp_folder` for storing temporary files,
    store features in `features_temp_folder`
    only perform calculation for the given `featurenames` (if such names are valid)
    if meta is true, also include mode0 features
    """
    msg_assert(featurenames is not None, "featurenames are required to be defined", f"featurenames ok")
    lInfo(f"handle : {video} for {modelname}")

    all_feat = all_no_ref_features()
    features_to_calculate, features = __filter_to_be_calculated_features(video, all_feat, featurenames, features_temp_folder)
    i = 0

    lInfo(f"calculate missing features {features_to_calculate} for {video}")
    if features_to_calculate != set():
        # convert to avpvs (rescale) and crop
        # assumes UHD-1/4K 60 fps video, yuv422p10le
        video_avpvs_crop = convert_to_avpvs_and_crop(
            video,
            f"{temp_folder}/crop/"
        )

        for frame in iterate_by_frame(video_avpvs_crop, convert=False):
            for f in features_to_calculate:
                x = features[f].calc(frame)
                lInfo(f"handle frame {i} of {video}: {f} -> {x}")
            i += 1
        os.remove(video_avpvs_crop)

    pooled_features, full_features = __store_and_pool_features(video, features, meta, features_temp_folder)
    return pooled_features, full_features


def extract_features_full_ref(dis_video, ref_video, temp_folder="./tmp", features_temp_folder="./tmp/features", featurenames=None, modelname="fume", meta=False):
    """
    extract full-reference features for a given dis_video and ref_video.
    use `temp_folder` for storing temporary files,
    store features in `features_temp_folder`
    only perform calculation for the given `featurenames` (if such names are valid)
    if meta is true, also include mode0 features
    """
    msg_assert(featurenames is not None, "featurenames are required to be defined", f"featurenames ok")
    lInfo(f"handle : {dis_video} for {modelname}")

    all_feat = all_features()
    features_to_calculate, features = __filter_to_be_calculated_features(dis_video, all_feat, featurenames, features_temp_folder)
    i = 0

    lInfo(f"calculate missing features {features_to_calculate} for {dis_video}, {ref_video}")
    if features_to_calculate != set():
        dis_basename = get_filename_without_extension(dis_video)

        # extract reference video properties
        ffprobe_res = ffprobe(ref_video)
        width = ffprobe_res["width"]
        height = ffprobe_res["height"]
        framerate = ffprobe_res["avg_frame_rate"]
        pix_fmt = ffprobe_res["pix_fmt"]

        lInfo(f"estimated src meta-data {width}x{height}@{framerate}:{pix_fmt}")
        dis_crop_folder = f"{temp_folder}/crop/{dis_basename}_dis/"
        ref_crop_folder = f"{temp_folder}/crop/{dis_basename}_ref/"
        # convert dis and ref video to to avpvs (rescale) and crop
        dis_video_avpvs_crop = convert_to_avpvs_and_crop(
            dis_video, dis_crop_folder,
            width=width,
            height=height,
            framerate=framerate,
            pix_fmt=pix_fmt
        )
        ref_video_avpvs_crop = convert_to_avpvs_and_crop(
            ref_video, ref_crop_folder,
            width=width,
            height=height,
            framerate=framerate,
            pix_fmt=pix_fmt
        )

        for d_frame, r_frame in iterate_by_frame_two_videos(dis_video_avpvs_crop, ref_video_avpvs_crop, convert=False):
            for f in features_to_calculate:
                x = features[f].calc_dis_ref(d_frame, r_frame)
                lInfo(f"handle frame {i} of {dis_video}: {f} -> {x}")
            i += 1

        # remove temp files
        shutil.rmtree(dis_crop_folder)
        shutil.rmtree(ref_crop_folder)
        # os.remove(dis_video_avpvs_crop)
        # os.remove(ref_video_avpvs_crop)

    pooled_features, full_features = __store_and_pool_features(dis_video, features, meta, features_temp_folder)
    return pooled_features, full_features


def predict_video_score(features, model_base_path, clipping=True):
    """
    based on the given features and model_base_path predict scores for all stored model types:
    - mos
    - classification
    - rating distribution
    further perform clipping if required and meaningful (e.g. in case of rating dist prediction no clipping is required)
    """
    df = pd.DataFrame([features])
    columns = df.columns.difference(["video", "src_video", "mos", "rating_dist"])
    X = df[sorted(columns)]
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0).values
    lInfo(f"loaded features {len(df)}: shape: {X.shape}")

    models = {
        "mos": model_base_path + "/model_regression.npz",
        "class": model_base_path + "/model_class.npz",
        "rating_dist": model_base_path + "/model_rating_dist.npz"
    }
    results = {}
    for m in models:
        if os.path.isfile(models[m]):
            lInfo(f"handle model {m}: {models[m]}")
            model = load_serialized(models[m])
            predicted = model.predict(X)
            # apply clipping if needed
            if clipping and m != "rating_dist":
                predicted = np.clip(predicted, 1, 5)
            # type conversion to float values
            predicted = [float(x) for x in predicted.flatten().tolist()]
            # some models have only one value, so just take this one value
            if len(predicted) == 1:
                predicted = predicted[0]
            results[m] = predicted
        else:
            lWarn(f"model {m} skipped, there is no trained model for this available, {models[m]}")
    results["model"] = model_base_path
    results["date"] = str(datetime.datetime.now())
    results["version"] = get_repo_version()
    return results