# FloorNet: A Unified Framework for Floorplan Reconstruction from 3D Scans
By Chen Liu<sup>\*</sup>, Jiaye Wu<sup>\*</sup>, and Yasutaka Furukawa (<sup>\*</sup> indicates equal contribution)

## Introduction

This paper proposes FloorNet, a novel neural network, to turn RGBD videos of indoor spaces into vector-graphics floorplans. FloorNet consists of three branches, PointNet branch, Floorplan branch, and Image branch. For more details, please refer to our ECCV 2018 [paper](https://arxiv.org/abs/1804.00090) or visit our [project website](http://art-programmer.github.io/floornet.html). This is a follow-up work of our floorplan transformation project which you can find [here](https://github.com/art-programmer/FloorplanTransformation).

## Updates
[12/22/2018] We now provide a free IP solver (not relying on Gurobi) at IP.py. The functionality of IP.py should be similar to QP.py which uses Gurobi to solve the IP problem. You might want to consider the free solver if you don't have a Gurobi license.

## Dependencies
Python 2.7, TensorFlow (>= 1.3), numpy, opencv 3, CUDA (>= 8.0), Gurobi (free only for academic usages).

## Data

### Dataset used in the paper

We collect 155 scans of residential units and annotated corresponding floorplan information. Among 155 scans, 135 are used for training and 20 are for testing. We convert data to tfrecords files which can be downloaded [here](https://drive.google.com/open?id=16lyX_xTiALUzKyst86WJHlhpTDr8XPF_) (or [here](https://mega.nz/#F!5yQy0b5T!ykkR4dqwGO9J5EwnKT_GBw) if you cannot access the previous one). Please put the downloaded files under folder *data/*.

Here are the links to the raw [point clouds](https://drive.google.com/open?id=1JJlD0qsgMpiU5Jq9TNm3uDPjvqi88aZn), [annotations](https://drive.google.com/open?id=1hYDE2SXLA8Cq7LEK67xO-UMeTSPJ5rcB), and their [associations](https://drive.google.com/open?id=125TAmYWk22EyzCdlbGIfX4Z4DRMhru_V). Please refer to **RecordWriterTango.py** to see how to convert the raw data and annotations to tfrecords files.


### Using custom data

To generate training/testing data from other data source, the data should be converted to tfrecords as what we did in **RecordWriterTango.py** (an example of our raw data before processed by **RecordWriterTango.py** is provided [here](https://mega.nz/#!dnohjKZa!I3NJZ806vNK-UYp-ap7OynGnS5E-E5AK_z5WsX8n1Ls)). Please refer to [this guide](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/) for how to generate and read tfrecords. 

Basically, every data sample(tf.train.Example) should at least contain the following components:


1. Inputs:

	- a point cloud (randomly sampled 50,000 points)
	- a mapping from point cloud's 3D space to 2D space of the 256x256 top-view density image.
		- It contains 50,000 indices, one for each point.
		- For point (x, y, z), index = round((y - min(Y) + padding) / (maxRange + 2 * padding) * 256) * 256 + round((x - min(X) + padding) / (maxRange + 2 * padding) * 256).
			- maxRange = max(max(X) - min(X), max(Y) - min(Y))
			- padding could be any small value, say 0.05 maxRange
	- optional: image features of the RGB video stream, if the image branch is enabled

2. Labels:

	- Corners and their corresponding types
	- Total number of corners
	- A ground-truth icon segmentation map
	- A ground-truth room segmentation map
	
Again, please refer to 	**RecordWriterTango.py** for exact details.

**NEW:** We added a template file, **RecordWriterCustom.py** for using custom data.


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
python train.py --task=evaluate --separateIconLoss
```

## Generate 3D models
We can popup the reconstructed floorplan to generate 3D models. Please refer to our previous project, [FloorplanTransformation](https://github.com/art-programmer/FloorplanTransformation), for more details.

## Contact

If you have any questions, please contact me at chenliu@wustl.edu.
