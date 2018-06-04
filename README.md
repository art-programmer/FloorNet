# FloorNet: A Unified Framework for Floorplan Reconstruction from 3D Scans
By Chen Liu<sup>\*</sup>, Jiaye Wu<sup>\*</sup>, and Yasutaka Furukawa (<sup>\*</sup> indicates equal contribution)

## Introduction

This paper proposes FloorNet, a novel neural network, to turn RGBD videos of indoor spaces into vector-graphics floorplans. FloorNet consists of three branches, PointNet branch, Floorplan branch, and Image branch. For more details, please refer to our Arxiv [paper](https://arxiv.org/abs/1804.00090) or visit our [project website](http://art-programmer.github.io/floornet.html). This is a follow-up work of our floorplan transformation project which you can find [here](https://github.com/art-programmer/FloorplanTransformation).

## Dependencies
Python 2.7, TensorFlow (>= 1.0), numpy, opencv 3.

### Data
We collect 155 scans of residential units and annotated corresponding floorplan information. Among 155 scans, 135 are used for training and 20 are for testing. We convert both training data and testing data to tfrecords files which can be downloaded [here](https://mega.nz/#F!5yQy0b5T!ykkR4dqwGO9J5EwnKT_GBw). Please put the downloaded files under folder *data/*.

### Training
To train the network from scratch, please run:
```bash
python train.py --restore=0
```

### Evaluation
To evaluate the performance of our trained model, please run:
```bash
python train.py --task=evaluate
```

## Contact

If you have any questions, please contact me at chenliu@wustl.edu.
