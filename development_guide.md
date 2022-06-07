# Development Guide

A detailed description of how the framework works internally and what the differences of the specific models are is described in the paper ["Modular Framework and Instances of Pixel-Based Video Quality Models for UHD-1/4K"](https://ieeexplore.ieee.org/document/9355144).
To speed up calculation, the framework uses a center crop of the videos, this decision was based on the observations made in the paper ["cencro - Speedup of Video Quality Calculation using Center Cropping"](https://www.researchgate.net/publication/338200687_cencro_--_Speedup_of_Video_Quality_Calculation_using_Center_Cropping).


## Extension of used features for a specific model

To extend the features for a specific model, you need to adjust in the `pixelmodels/common.py` the following methods, depending on the feature type:

```python
# for no reference features
def all_no_ref_features():
    ...
    return {
        "contrast": ImageFeature(calc_contrast_features),
        ...
        "NEW_FEATURE": INSTANCE OF THE FEATURE
        ...
    }

# and similar you could extend specific full-reference features in:
def all_features():
    ...

```


These both of these methdos are then later used to map the features to the corresponding instances, while the 
models are defined.
The "INSTANCE OF THE FEATURE" must be of type `quat.video.base_features.Feature`, thus it must implement the required methods of this class.

Afterwards, you can adjust for a specific model the used features by extending the methods defined for each model, e.g. for `pixelmodels/nofu.py`:

```python
def nofu_features():
    return {
        "contrast",
        ...
        "NEW_FEATURE"
        ...
    }
```

The definition in this file is only based on the name, have in mind that these names must be uniqe, and they will be used for caching as filenames (so please do not use any fancy characters, to be sure just use ASCII chars).

**HINT** an extension of the used features always requires a full re-training of the models (if you have performed a training before only the new features are calculated)

