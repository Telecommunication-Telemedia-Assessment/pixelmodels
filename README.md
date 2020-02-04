# Pixel based Video Quality Models for 4K

The following repository consists of in total 4 pixel based video qualtiy models.
Including no-reference, full-reference and for each a hybrid mode 0 based version.

If you use code, or the models in any reseach please cite the following paper

```
@inproceedings{goering2020pixel,
  title={Pixel based Video Quality Models for 4K},
  author={Steve G{\"{o}}ring and Rakesh Rao Ramachandra Rao and Bernhard Feiten and Alexander Raake},
  year={2020},
  organization={IEEE},
}
```

## Requirements
The models and software is only tested on linux systems (e.g. Ubuntu 19.04)

* python3, python3-pip, python3-venv
* poetry >= 1.0.3 (e.g. pip3 install poetry==1.0.3)
* ffmpeg
* git

Run the following command:

```bash
poetry install
```
(if you have problems with pip, run `pip3 install --user -U pip`)

## Included video quality models
In total in this repository four video quality models are included:

* nofu: no-reference pixel based model
* hyfu: hybrid no-reference pixel based model
* fume: full-reference pixel based model
* hyfr: hybrid full-reference model

Both hybrid models require access to bitrate, resolution, codec and framerate.
This meta-data will be automatically extracted from the given video files.

### Usage nofu

To use the provided tool, e.g. run
```bash
poetry run nofu test_videos/test_video_h264.mkv
```

### Retraining the models

To retrain the models it is required to have CSV files accoding to the used format of [AVT-VQDB-UHD-1](https://github.com/Telecommunication-Telemedia-Assessment/AVT-VQDB-UHD-1)

To enable the rating distribution training additional data is requred, that is.e.g not part of the AVT-VQDB-UHD-1 dataset.

For each model a train_{modelname} tool is provided that can be started, e.g. for nofu with the following command line:
```bash
poetry run train_nofu data/AVT-VQDB-UHD-1/test_1/mos_ci.csv
```