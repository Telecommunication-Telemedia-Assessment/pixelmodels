# pixel_based_4k_video_quality_models

all models in one

## Requirements

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

Both hybrid models require access to  bitrate, resolution, codec, framerate

### Usage nofu

To use the provided tool, e.g. run
```bash
poetry run nofu test_videos/test_video_h264.mkv
```
