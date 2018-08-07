# FloorNet: A Unified Framework for Floorplan Reconstruction from 3D Scans
By Chen Liu<sup>\*</sup>, Jiaye Wu<sup>\*</sup>, and Yasutaka Furukawa (<sup>\*</sup> indicates equal contribution)

## Introduction

This paper proposes FloorNet, a novel neural network, to turn RGBD videos of indoor spaces into vector-graphics floorplans. FloorNet consists of three branches, PointNet branch, Floorplan branch, and Image branch. For more details, please refer to our ECCV 2018 [paper](https://arxiv.org/abs/1804.00090) or visit our [project website](http://art-programmer.github.io/floornet.html). This is a follow-up work of our floorplan transformation project which you can find [here](https://github.com/art-programmer/FloorplanTransformation).

## Dependencies
Python 2.7, TensorFlow (>= 1.3), numpy, opencv 3.

## Data

### Dataset used in the paper

We collect 155 scans of residential units and annotated corresponding floorplan information. Among 155 scans, 135 are used for training and 20 are for testing. We convert both training data and testing data to [tfrecords files](https://www.tensorflow.org/guide/datasets#consuming_tfrecord_data) which can be downloaded [here](https://mega.nz/#F!5yQy0b5T!ykkR4dqwGO9J5EwnKT_GBw). Please put the downloaded files under folder *data/*.


### Using other data

To generate training/testing data from other data source, the data should be converted to tfrecords as what we did in **RecordWriterTango.py**. Please refer to [this guide](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/) for how to generate and read tfrecords. 

Basically, every data sample(tf.train.Example) should at least contain the following components:


1. Inputs:

	- a point cloud(after filtering to 50,000 points), 
	- a mapping from point cloud's 3D space to 2D space of the top-view density image
	- optional: image features of the RGB video stream, if the image branch is enabled

2. Labels:

	- Corners and their corresponding types
	- Total number of corners
	- A ground-truth icon segmentation map
	- A ground-truth room segmentation map
	
Again, please refer to 	**RecordWriterTango.py** for exact details.


## Annotator
For reference, a similar (but not the same) annotator written in Python is [here](https://github.com/art-programmer/FloorplanAnnotator). You need to make some changes to annotate your own data.

## Training
To train the network from scratch, please run:
```bash
python train.py --restore=0
```

## Evaluation
To evaluate the performance of our trained model, please run:
```bash
python train.py --task=evaluate
```

## Generate 3D models
We can popup the reconstructed floorplan to generate 3D models. Please refer to our previous project, [FloorplanTransformation](https://github.com/art-programmer/FloorplanTransformation), for more details.

## Contact

If you have any questions, please contact me at chenliu@wustl.edu.
