# The Attribution Bottleneck

This is the source code for the paper "Information Bottleneck for Attribution:
What is Sufficient for Prediction?".

## Setup

1. Clone this repository:
    ```
    $ git clone [URL] && cd attribution-bottleneck-pytorch
    ```
2. Create a conda environment with all packages:
   ```
   $ conda create -n new environment --file requirements.txt
   ```

3. Using your new conda environment, install this repository with pip:
   ```
   $ pip install .
    ```

4. Download the model weights from the [release page](releases) and unpack them
   in the repository root directory:
   ```
   $ tar -xvf bottleneck_for_attribution_weights.tar.gz
   ```

Optional:


5. If you want to retrain the Readout Bottleneck, place the imagenet dataset under `data/imagenet`. You might just create
   a link with `ln -s [image dir]  data/imagenet`.

6. Test it with:
   ```
   $ python ./scripts/eval_degradation.py resnet50 8 Saliency test
   ```

## Usage

We provide some jupyter notebooks to demonstrate the usage of both per-sample and readout bottleneck.
* `example_per-sample.ipynb` : Usage of the Per-Sample Bottleneck on an example image
* `example_readout.ipynb` : Usage of the Readout Bottleneck on an example image
* `compare_methods.ipynb` : Visually compare different attribution methods on an example image

## Scripts

The scripts to reproduce our evaluation can be found in the [scripts
directory](scripts).
Following attributions are implemented:



For the bounding box task, replace the model with either `vgg16` or `resnet50`.
```bash
$eval_bounding_boxes.py [model] [attribution]
```

For the degradation task, you also have specify the tile size. In the paper, we
used `8` and `14`.
```bash
$ eval_degradation.py [model] [tile size] [attribution]
```

The results on sensitivity-n can be calculated with:
```bash
eval_sensitivity_n.py [model] [tile size] [attribution]
```

