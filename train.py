import multiprocessing as mp
import sys
import tensorflow as tf
import numpy as np
import cv2
import tflearn
import random
import math
import os
import time
import zlib
import socket
import traceback
try:
    import Queue
except Exception:
    import queue as Queue
import sys
#import tf_nndistance
try:
    import cPickle as pickle
except:
    import pickle
import argparse

from RecordReader import *
from utils import *
import traceback
import matplotlib.pyplot as plt
if 'xrange' not in globals():
    xrange = range
    pass
import tensorflow.contrib.slim as slim
from tensorflow import nn
from tensorflow.python.client import timeline
#from evaluate import *

bn_func = tflearn.layers.normalization.batch_normalization

DATASETS = ['syn', 'real', 'scannet', "matterport", "SUNCG"]

HEATMAP_SCALE = 3


def mergeFeatures(features):
    if True:
        return tf.add_n(features)
    else:
        return tf.concat(features, axis=-1)
        pass

def dumpOutputs(corners, semantics=None):
    corners = sigmoid(corners)
    corners = np.transpose(corners, [0, 3, 1, 2])
    #print(corners.shape)
    np.save('output/corners.npy', corners)
    return



def build_graph_pointnet(options, input_dict):
    nChannels = [7, 64, 64, 64, 128, 1024]
    sizes = [HEIGHT, HEIGHT // 2, HEIGHT // 4, HEIGHT // 8, HEIGHT // 16, HEIGHT // 32]

    with tf.device('/gpu:%s'%options.gpu_id[0]):
        tflearn.config.init_training_mode()


        #tflearn.init_graph(seed=1029,num_cores=2,gpu_memory_fraction=1.0,soft_placement=True)
        #tf.set_random_seed(1029)
        pointcloud_inp = input_dict['points']
        pointcloud_indices_inp = input_dict['point_indices']
        pointcloud_indices_inp += tf.expand_dims(tf.range(options.batchSize) * sizes[0] * sizes[0], -1)
        # batchIndexOffsets = []
        # for c in xrange(6):
        #     batchIndexOffsets.append(tf.range(options.batchSize) * sizes[c] * sizes[c])
        #     continue
        # batchIndexOffsets = tf.expand_dims(tf.stack(batchIndexOffsets, axis=1), -1)

        # indices_maps = tf.unstack(pointcloud_indices_inp + batchIndexOffsets, axis=1)

        x0 = tf.expand_dims(pointcloud_inp, -1)
        x1 = slim.conv2d(x0, nChannels[1], (1, nChannels[0]), stride=1, activation_fn=nn.relu, padding='valid', normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

        x2 = slim.conv2d(x1, nChannels[2], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

        x3 = slim.conv2d(x2, nChannels[3], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

        x4 = slim.conv2d(x3, nChannels[4], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

        x5 = slim.conv2d(x4, nChannels[5], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

        x_fc_topdown = tflearn.layers.max_pool_2d(x5, (options.numPoints, 1), strides=1, padding='valid')

        x_fc = x_fc_topdown

        # x_fc = tf.reshape(x_fc, (options.batchSize, -1))
        # x_fc = slim.fully_connected(x_fc, 256, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
        # x_fc = slim.fully_connected(x_fc, nChannels[5], activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
        # x_fc = tf.reshape(x_fc, (options.batchSize, 1, 1, -1))

        x_fc_up = tf.tile(x_fc, (1, options.numPoints, 1, 1))

        x_fc_up = tf.concat([x_fc_up, x3], axis=-1)

        x5_up = slim.conv2d(x_fc_up, 512, (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
        x4_up = slim.conv2d(x5_up, 256, (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
        x3_up = slim.conv2d(x4_up, 128, (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
        x2_up = slim.conv2d(x3_up, 128, (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

        #x2_up = tf.unsorted_segment_sum(tf.reshape(x2_up, (-1, 128)), tf.reshape(pointcloud_indices_inp, (-1, )), num_segments=options.batchSize * sizes[0] * sizes[0]) / options.sumScale
        x2_up = tf.maximum(tf.unsorted_segment_max(tf.reshape(x2_up, (-1, 128)), tf.reshape(pointcloud_indices_inp, (-1, )), num_segments=options.batchSize * sizes[0] * sizes[0], name="project"), 0)
        x2_up = tf.reshape(x2_up, (options.batchSize, sizes[0], sizes[0], -1))
        x1_up = x2_up
        #x1_up = slim.conv2d(x2_up, nChannels[1], (3, 3), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
        #x0_topdown = tf.reshape(x0_topdown, (options.batchSize, sizes[0], sizes[0], -1))

        if False:
            pred_corner = slim.conv2d(x1_up, 64, [3, 3], stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_corner')
            pred_icon = slim.conv2d(x1_up, 64, [3, 3], stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_icon')
            pred_room = slim.conv2d(x1_up, 64, [3, 3], stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_room')

            pred_corner = slim.conv2d(pred_corner, NUM_CORNERS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
            pred_icon = slim.conv2d(pred_icon, NUM_ICONS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
            pred_room = slim.conv2d(pred_room, NUM_ROOMS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
        else:
            pred_corner = slim.conv2d(x1_up, NUM_CORNERS, [3, 3], stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_corner')
            pred_icon = slim.conv2d(x1_up, NUM_ICONS, [3, 3], stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_icon')
            pred_room = slim.conv2d(x1_up, NUM_ROOMS, [3, 3], stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_room')
            pass

        pred_dict = {'corner': pred_corner, 'icon': pred_icon, 'room': pred_room}
        x0_topdown = tf.unsorted_segment_sum(tf.reshape(x0, (-1, NUM_CHANNELS[0])), tf.reshape(pointcloud_indices_inp, (-1, )), num_segments=options.batchSize / len(options.gpu_id) * SIZES[0] * SIZES[0]) / options.sumScale
        x0_topdown = tf.reshape(x0_topdown, (options.batchSize/len(options.gpu_id), SIZES[0], SIZES[0], -1))
        debug_dict = {'x0_topdown': x0_topdown}
        pass

    return pred_dict, debug_dict

def build_graph_image(options, input_dict):
    image_features_all = input_dict['image_features']
    if len(options.gpu_id) > 1:
        img_features = [tf.split(features, len(options.gpu_id), axis=0) for k, features in image_features_all.iteritems()]
    else:
        img_features = [[features for k, features in image_features_all.iteritems()]]
        pass
    #pointcloud_inp = tf.placeholder(tf.float32,shape=(options.batchSize, options.numPoints, options.numInputChannels),name='pointcloud_inp')
    #pointcloud_indices_inp = tf.placeholder(tf.int32,shape=(options.batchSize, 6, options.numPoints),name='pointcloud_indices_inp')

    pred_dicts = []
    debug_dicts = []

    reused = False
    for i, img_feature in zip(options.gpu_id, img_features):
        with tf.device('/gpu:%s'%int(i)), tf.variable_scope('floorplan_net', reuse=reused), slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):
            x5_up = img_feature[4]

            x4_up = slim.conv2d_transpose(x5_up, NUM_CHANNELS[4], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

            x4_up = mergeFeatures([x4_up, img_feature[3]])

            x3_up = slim.conv2d_transpose(x4_up, NUM_CHANNELS[3], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
            #print(x3_up)

            x3_up = mergeFeatures([x3_up, img_feature[2]])

            x2_up = slim.conv2d_transpose(x3_up, NUM_CHANNELS[2], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

            x2_up = mergeFeatures([x2_up, img_feature[1]])

            x1_up = slim.conv2d_transpose(x2_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

            x1_up = mergeFeatures([x1_up, img_feature[0]])

            if True:
                pred_corner = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_corner')
                pred_icon = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_icon')
                pred_room = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_room')

                pred_corner = slim.conv2d(pred_corner, NUM_CORNERS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
                pred_icon = slim.conv2d(pred_icon, NUM_ICONS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
                pred_room = slim.conv2d(pred_room, NUM_ROOMS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
            else:
                pred_corner = slim.conv2d_transpose(x1_up, NUM_CORNERS, [5, 5], stride=2, activation_fn=None, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_corner')
                pred_icon = slim.conv2d_transpose(x1_up, NUM_ICONS, [5, 5], stride=2, activation_fn=None, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_icon')
                pred_room = slim.conv2d_transpose(x1_up, NUM_ROOMS, [5, 5], stride=2, activation_fn=None, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_room')
                pass

            pred_dict = {'corner': pred_corner, 'icon': pred_icon, 'room': pred_room}
            debug_dict = {'x0_topdown': tf.zeros((options.batchSize, HEIGHT, WIDTH, options.numInputChannels))}

            pred_dicts.append(pred_dict)
            debug_dicts.append(debug_dict)
            pass
        continue

    pred_dict = pred_dicts[0]
    for pred in pred_dicts[1:]:
        for k,v in pred.items():
            pred_dict[k] = tf.concat([pred_dict[k], v], axis=0)
    debug_dict = debug_dicts[0]
    for de in debug_dicts[1:]:
        for k,v in de.items():
            debug_dict[k] = tf.concat([debug_dict[k], v], axis=0)

    return pred_dict, debug_dict


def build_graph(options, input_dict):

    branches_options = set(options.branches)
    simple_option = set(['0', '6', '7'])
    if (simple_option | branches_options) == simple_option:
        return build_graph_pointnet(options, input_dict)

    if options.branches == '4':
        return build_graph_image(options, input_dict)

    pointcloud_inp_all = input_dict['points']
    pointcloud_indices_inp_all = input_dict['point_indices']


    if len(options.gpu_id) > 1:
        pointcloud_inps = tf.split(pointcloud_inp_all, len(options.gpu_id), axis=0)
        pointcloud_indices_inps = tf.split(pointcloud_indices_inp_all, len(options.gpu_id), axis=0)
    else:
        pointcloud_inps = [pointcloud_inp_all]
        pointcloud_indices_inps = [pointcloud_indices_inp_all]
        pass
    #print(pointcloud_inps)
    #print(pointcloud_inps[0])
    if '4' in options.branches:
        image_features_all = input_dict['image_features']
        if len(options.gpu_id) > 1:
            img_features = [tf.split(features, len(options.gpu_id), axis=0) for k, features in image_features_all.iteritems()]
        else:
            img_features = [[features for k, features in image_features_all.iteritems()]]
            pass
    else:
        img_features = [None for _ in xrange(len(options.gpu_id))]
        pass

    #pointcloud_inp = tf.placeholder(tf.float32,shape=(options.batchSize, options.numPoints, options.numInputChannels),name='pointcloud_inp')
    #pointcloud_indices_inp = tf.placeholder(tf.int32,shape=(options.batchSize, 6, options.numPoints),name='pointcloud_indices_inp')

    pred_dicts = []
    debug_dicts = []
    reused = False
    for i, pointcloud_inp, pointcloud_indices_inp, img_feature in zip(options.gpu_id, pointcloud_inps, pointcloud_indices_inps, img_features):
        #for i in range(1):
        with tf.device('/gpu:%s'%int(i)), tf.variable_scope('floorplan_net', reuse=reused), slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

    # if True:
    #     pointcloud_inp = input_dict['points']
    #     pointcloud_indices_inp = input_dict['point_indices']
    #     if '4' in options.branches:
    #         img_feature = input_dict['image_features']
    #         pass
    #     with tf.device('/gpu:0'):

            reused=True
            debug_dict = {}
            tf.set_random_seed(1029)
            #tflearn.init_graph(seed=1029,num_cores=2,gpu_memory_fraction=1.0,soft_placement=False, log_device=True)

            if 'd' in options.augmentation:
                #keep_prob = tf.random_uniform([1], minval=0.5, maxval=1.0)[0]
                keep_prob = 0.5
                pointcloud_inp = tf.nn.dropout(pointcloud_inp, keep_prob, noise_shape=[options.batchSize, NUM_POINTS, 1]) * keep_prob
                pointcloud_indices_inp = tf.cast(tf.round(tf.nn.dropout(tf.cast(pointcloud_indices_inp, np.float32), keep_prob) * keep_prob), tf.int32)
                pass

            pointcloud_indices_inp = getCoarseIndicesMapsBatch(pointcloud_indices_inp, WIDTH, HEIGHT)
            batchIndexOffsets = []
            for c in xrange(6):
                batchIndexOffsets.append((np.arange(options.batchSize/len(options.gpu_id), dtype=np.int32)) * SIZES[c] * SIZES[c])
                continue
            batchIndexOffsets = tf.expand_dims(tf.stack(batchIndexOffsets, axis=0), -1)

            #print(pointcloud_indices_inp, batchIndexOffsets)
            #exit(1)

            indices_maps = pointcloud_indices_inp + batchIndexOffsets


            x0 = tf.expand_dims(pointcloud_inp, -1)

            x0_topdown = tf.unsorted_segment_sum(tf.reshape(x0, (-1, NUM_CHANNELS[0])), tf.reshape(indices_maps[0], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[0] * SIZES[0]) / options.sumScale
            x0_topdown = tf.reshape(x0_topdown, (options.batchSize/len(options.gpu_id), SIZES[0], SIZES[0], -1))
            x0_down = x0_topdown

            if options.branches == '1':
                x1_down = slim.conv2d(x0_down, NUM_CHANNELS[1], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                x2_down = slim.conv2d(x1_down, NUM_CHANNELS[2], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                x3_down = slim.conv2d(x2_down, NUM_CHANNELS[3], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                x4_down = slim.conv2d(x3_down, NUM_CHANNELS[4], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                x5_down = slim.conv2d(x4_down, NUM_CHANNELS[5], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                x5_up = x5_down
            else:
                x1 = slim.conv2d(x0, NUM_CHANNELS[1], (1, NUM_CHANNELS[0]), stride=1, activation_fn=nn.relu, padding='valid', normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if options.poolingTypes[0] == 's':
                    x1_topdown = tf.unsorted_segment_sum(tf.reshape(x1, (-1, NUM_CHANNELS[1])), tf.reshape(indices_maps[1], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[1] * SIZES[1]) / min(options.sumScale * 4, (options.sumScale - 1) * 10000 + 1)
                else:
                    x1_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x1, (-1, NUM_CHANNELS[1])), tf.reshape(indices_maps[1], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[1] * SIZES[1]), 0)
                    pass

                x1_topdown = tf.reshape(x1_topdown, (options.batchSize/len(options.gpu_id), SIZES[1], SIZES[1], -1))
                x1_down = slim.conv2d(x0_down, NUM_CHANNELS[1], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if '2' in options.branches:
                    x1_unproject = tf.reshape(tf.gather(tf.reshape(x1_down, (-1, NUM_CHANNELS[1])), indices_maps[1], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    x1 = mergeFeatures([x1, tf.expand_dims(x1_unproject, 2)])
                    pass

                if '0' in options.branches:
                    x1_down = mergeFeatures([x1_topdown, x1_down])
                    pass


                x2 = slim.conv2d(x1, NUM_CHANNELS[2], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if options.poolingTypes[1] == 's':
                    x2_topdown = tf.unsorted_segment_sum(tf.reshape(x2, (-1, NUM_CHANNELS[2])), tf.reshape(indices_maps[2], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[2] * SIZES[2]) / min(options.sumScale * 16, (options.sumScale - 1) * 10000 + 1)
                else:
                    x2_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x2, (-1, NUM_CHANNELS[2])), tf.reshape(indices_maps[2], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[2] * SIZES[2]), 0)
                    pass

                x2_topdown = tf.reshape(x2_topdown, (options.batchSize/len(options.gpu_id), SIZES[2], SIZES[2], -1))
                x2_down = slim.conv2d(x1_down, NUM_CHANNELS[2], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if '2' in options.branches:
                    x2_unproject = tf.reshape(tf.gather(tf.reshape(x2_down, (-1, NUM_CHANNELS[2])), indices_maps[2], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    x2 = mergeFeatures([x2, tf.expand_dims(x2_unproject, 2)])
                    pass
                if '0' in options.branches:
                    x2_down = mergeFeatures([x2_topdown, x2_down])
                    pass


                x3 = slim.conv2d(x2, NUM_CHANNELS[3], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if options.poolingTypes[2] == 's':
                    x3_topdown = tf.unsorted_segment_sum(tf.reshape(x3, (-1, NUM_CHANNELS[3])), tf.reshape(indices_maps[3], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[3] * SIZES[3]) / min(options.sumScale * 64, (options.sumScale - 1) * 10000 + 1)
                else:
                    x3_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x3, (-1, NUM_CHANNELS[3])), tf.reshape(indices_maps[3], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[3] * SIZES[3]), 0)
                    pass

                x3_topdown = tf.reshape(x3_topdown, (options.batchSize/len(options.gpu_id), SIZES[3], SIZES[3], -1))
                x3_down = slim.conv2d(x2_down, NUM_CHANNELS[3], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if '2' in options.branches:
                    x3_unproject = tf.reshape(tf.gather(tf.reshape(x3_down, (-1, NUM_CHANNELS[3])), indices_maps[3], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    x3 = mergeFeatures([x3, tf.expand_dims(x3_unproject, 2)])
                    pass

                if '0' in options.branches:
                    x3_down = mergeFeatures([x3_topdown, x3_down])
                    pass


                x4 = slim.conv2d(x3, NUM_CHANNELS[4], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if options.poolingTypes[3] == 's':
                     x4_topdown = tf.unsorted_segment_sum(tf.reshape(x4, (-1, NUM_CHANNELS[4])), tf.reshape(indices_maps[4], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[4] * SIZES[4]) / min(options.sumScale * 256, (options.sumScale - 1) * 10000 + 1)
                else:
                    x4_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x4, (-1, NUM_CHANNELS[4])), tf.reshape(indices_maps[4], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[4] * SIZES[4]), 0)
                    pass
                x4_topdown = tf.reshape(x4_topdown, (options.batchSize/len(options.gpu_id), SIZES[4], SIZES[4], -1))
                x4_down = slim.conv2d(x3_down, NUM_CHANNELS[4], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if '2' in options.branches:
                    x4_unproject = tf.reshape(tf.gather(tf.reshape(x4_down, (-1, NUM_CHANNELS[4])), indices_maps[4], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    x4 = mergeFeatures([x4, tf.expand_dims(x4_unproject, 2)])
                    pass
                if '0' in options.branches:
                    x4_down = mergeFeatures([x4_topdown, x4_down])
                    pass


                x5 = slim.conv2d(x4, NUM_CHANNELS[5], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if options.poolingTypes[4] == 's':
                    x5_topdown = tf.unsorted_segment_sum(tf.reshape(x5, (-1, NUM_CHANNELS[5])), tf.reshape(indices_maps[5], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[5] * SIZES[5]) / min(options.sumScale * 1024, (options.sumScale - 1) * 10000 + 1)
                else:
                    x5_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x5, (-1, NUM_CHANNELS[5])), tf.reshape(indices_maps[5], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[5] * SIZES[5]), 0)
                    pass
                x5_topdown = tf.reshape(x5_topdown, (options.batchSize/len(options.gpu_id), SIZES[5], SIZES[5], -1))
                x5_down = slim.conv2d(x4_down, NUM_CHANNELS[5], (3, 3), stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if '2' in options.branches:
                    x5_unproject = tf.reshape(tf.gather(tf.reshape(x5_down, (-1, NUM_CHANNELS[5])), indices_maps[5], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    x5 = mergeFeatures([x5, tf.expand_dims(x5_unproject, 2)])
                    pass
                if '0' in options.branches:
                    x5_down = mergeFeatures([x5_topdown, x5_down])
                    pass


                x_fc_topdown = tflearn.layers.max_pool_2d(x5, (options.numPoints, 1), strides=1, padding='valid')

                x_fc_down = tflearn.layers.max_pool_2d(x5_down, (SIZES[5], SIZES[5]), strides=1, padding='valid')
                if '0' in options.branches:
                    x_fc_down = mergeFeatures([x_fc_topdown, x_fc_down])
                    pass

                # no fully connected layer in the middle
                #x_fc = tf.reshape(x_fc, (options.batchSize/len(options.gpu_id), -1))
                #x_fc = slim.fully_connected(x_fc, 256, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                #x_fc = slim.fully_connected(x_fc, NUM_CHANNELS[5], activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                #x_fc = tf.reshape(x_fc, (options.batchSize/len(options.gpu_id), 1, 1, -1))

                x_fc_up = tf.tile(x_fc_down, (1, SIZES[5], SIZES[5], 1))
                x5_up = mergeFeatures([x_fc_up, x5_down])
                pass


            if '3' in options.branches:
                if '0' in options.branches:
                    x_fc = tf.tile(x_fc_down, (1, options.numPoints, 1, 1))
                else:
                    x_fc = tf.tile(x_fc_topdown, (1, options.numPoints, 1, 1))
                    pass
                # merge local point features and global point features
                x_fc = tf.concat([x_fc, x3], axis=-1)
                x5 = slim.conv2d(x_fc, NUM_CHANNELS[5], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

                pass
            if '4' in options.branches:
                x5_up = mergeFeatures([x5_up, img_feature[4]])
                pass


            x4_up = slim.conv2d_transpose(x5_up, NUM_CHANNELS[4], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
            x4_up = mergeFeatures([x4_up, x4_down])

            if '3' in options.branches:
                x4 = slim.conv2d(x5, NUM_CHANNELS[4], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
                if '2' in options.branches:
                    x4_up_unproject = tf.reshape(tf.gather(tf.reshape(x4_up, (-1, NUM_CHANNELS[4])), indices_maps[4], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    #x4 = tf.add(x4, tf.expand_dims(x4_up_unproject, 2))
                    x4 = mergeFeatures([x4, tf.expand_dims(x4_up_unproject, 2)])
                    pass

                x4_up_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x4, (-1, NUM_CHANNELS[4])), tf.reshape(indices_maps[4], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[4] * SIZES[4]), 0)
                x4_up_topdown = tf.reshape(x4_up_topdown, (options.batchSize/len(options.gpu_id), SIZES[4], SIZES[4], -1))
                if '0' in options.branches:
                    x4_up = mergeFeatures([x4_up, x4_up_topdown])
                    pass
                pass
            if '4' in options.branches:
                x4_up = mergeFeatures([x4_up, img_feature[3]])
                pass


            x3_up = slim.conv2d_transpose(x4_up, NUM_CHANNELS[3], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
            #print(x3_up)
            x3_up = mergeFeatures([x3_up, x3_down])

            # if '4' in options.branches:
            #     img_SIZES = tf.constant([32, 32], dtype='int32')
            #     resized_image = tf.image.resize_images(img_feature, img_SIZES)
            #     x3_up = mergeFeatures([x3_up, resized_image])
            #     pass

            if '3' in options.branches:
                x3 = slim.conv2d(x4, NUM_CHANNELS[3], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

                if '2' in options.branches:
                    x3_up_unproject = tf.reshape(tf.gather(tf.reshape(x3_up, (-1, NUM_CHANNELS[3])), indices_maps[3], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    #x3 = tf.add(x3, tf.expand_dims(x3_up_unproject, 2))
                    x3 = mergeFeatures([x3, tf.expand_dims(x3_up_unproject, 2)])
                    pass

                x3_up_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x3, (-1, NUM_CHANNELS[3])), tf.reshape(indices_maps[3], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[3] * SIZES[3]), 0)
                x3_up_topdown = tf.reshape(x3_up_topdown, (options.batchSize/len(options.gpu_id), SIZES[3], SIZES[3], -1))
                if '0' in options.branches:
                    x3_up = mergeFeatures([x3_up, x3_up_topdown])
                    pass
                pass
            if '4' in options.branches:
                x3_up = mergeFeatures([x3_up, img_feature[2]])
                pass


            x2_up = slim.conv2d_transpose(x3_up, NUM_CHANNELS[2], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
            x2_up = mergeFeatures([x2_up, x2_down])

            if '3' in options.branches:
                x2 = slim.conv2d(x3, NUM_CHANNELS[2], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

                if '2' in options.branches:
                    x2_up_unproject = tf.reshape(tf.gather(tf.reshape(x2_up, (-1, NUM_CHANNELS[2])), indices_maps[2], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    #x2 = tf.add(x2, tf.expand_dims(x2_up_unproject, 2))
                    x2 = mergeFeatures([x2, tf.expand_dims(x2_up_unproject, 2)])
                    pass

                x2_up_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x2, (-1, NUM_CHANNELS[2])), tf.reshape(indices_maps[2], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[2] * SIZES[2]), 0)
                x2_up_topdown = tf.reshape(x2_up_topdown, (options.batchSize/len(options.gpu_id), SIZES[2], SIZES[2], -1))
                if '0' in options.branches:
                    x2_up = mergeFeatures([x2_up, x2_up_topdown])
                    pass
                pass
            if '4' in options.branches:
                x2_up = mergeFeatures([x2_up, img_feature[1]])
                pass


            x1_up = slim.conv2d_transpose(x2_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))
            #print(x1_up.shape)
            #print(x1_down.shape)
            x1_up = mergeFeatures([x1_up, x1_down])

            if '3' in options.branches:
                x1 = slim.conv2d(x2, NUM_CHANNELS[1], (1, 1), stride=1, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5))

                if '2' in options.branches:
                    x1_up_unproject = tf.reshape(tf.gather(tf.reshape(x1_up, (-1, NUM_CHANNELS[1])), indices_maps[1], validate_indices=False), (options.batchSize/len(options.gpu_id), options.numPoints, -1))
                    #print(x1_up_unproject.shape)
                    #x1 = tf.add(x1, tf.expand_dims(x1_up_unproject, 2))
                    x1 = mergeFeatures([x1, tf.expand_dims(x1_up_unproject, 2)])
                    pass

                x1_up_topdown = tf.maximum(tf.unsorted_segment_max(tf.reshape(x1, (-1, NUM_CHANNELS[1])), tf.reshape(indices_maps[1], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[1] * SIZES[1]), 0)
                x1_up_topdown = tf.reshape(x1_up_topdown, (options.batchSize/len(options.gpu_id), SIZES[1], SIZES[1], -1))
                if '1' not in options.branches:
                    x1_up = x1_up_topdown
                else:
                    #print(x1_up_topdown)
                    x1_up = mergeFeatures([x1_up, x1_up_topdown])
                    pass
                pass
            if '4' in options.branches:
                x1_up = mergeFeatures([x1_up, img_feature[0]])
                pass

            #print(x1_up)
            #print(NUM_ROOMS)
            if options.outputLayers in ['two', 'nobn']:
                if options.outputLayers == 'two':
                    pred_corner = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_corner')
                    pred_icon = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_icon')
                    pred_room = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_room')
                else:
                    pred_corner = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_corner')
                    pred_icon = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_icon')
                    pred_room = slim.conv2d_transpose(x1_up, NUM_CHANNELS[1], [5, 5], stride=2, activation_fn=nn.relu, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_room')
                    pass

                pred_corner = slim.conv2d(pred_corner, NUM_CORNERS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
                pred_icon = slim.conv2d(pred_icon, NUM_ICONS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
                pred_room = slim.conv2d(pred_room, NUM_ROOMS, (1, 1), stride=1, activation_fn=None, normalizer_fn=None, weights_regularizer=slim.l2_regularizer(1e-5))
            else:
                pred_corner = slim.conv2d_transpose(x1_up, NUM_CORNERS, [5, 5], stride=2, activation_fn=None, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_corner')
                pred_icon = slim.conv2d_transpose(x1_up, NUM_ICONS, [5, 5], stride=2, activation_fn=None, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_icon')
                pred_room = slim.conv2d_transpose(x1_up, NUM_ROOMS, [5, 5], stride=2, activation_fn=None, normalizer_fn=bn_func, weights_regularizer=slim.l2_regularizer(1e-5), scope='pred_room')
                pass


            pred_dict = {'corner': pred_corner, 'icon': pred_icon, 'room': pred_room}
            debug_dict['x0_topdown'] = x0_topdown

            # x1_topdown = tf.unsorted_segment_sum(tf.reshape(x0, (-1, NUM_CHANNELS[0])), tf.reshape(indices_maps[1], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[1] * SIZES[1]) / (options.sumScale * 4)
            # x1_topdown = tf.reshape(x1_topdown, (options.batchSize/len(options.gpu_id), SIZES[1], SIZES[1], -1))
            # debug_dict['x1_topdown'] = x1_topdown

            # x2_topdown = tf.unsorted_segment_sum(tf.reshape(x0, (-1, NUM_CHANNELS[0])), tf.reshape(indices_maps[2], (-1, )), num_segments=options.batchSize/len(options.gpu_id) * SIZES[2] * SIZES[2]) / (options.sumScale * 16)
            # x2_topdown = tf.reshape(x2_topdown, (options.batchSize/len(options.gpu_id), SIZES[2], SIZES[2], -1))
            # debug_dict['x2_topdown'] = x2_topdown
            # debug_dict['x1_up'] = x1_up
            #debug_dict['resized_image'] = resized_image

            pred_dicts.append(pred_dict)
            debug_dicts.append(debug_dict)
            pass
        continue

    pred_dict = pred_dicts[0]
    for pred in pred_dicts[1:]:
        for k,v in pred.items():
            pred_dict[k] = tf.concat([pred_dict[k], v], axis=0)
    debug_dict = debug_dicts[0]
    for de in debug_dicts[1:]:
        for k,v in de.items():
            debug_dict[k] = tf.concat([debug_dict[k], v], axis=0)

    return pred_dict, debug_dict


def build_loss(options, pred_dict, gt_dict, dataset_flag, debug_dict, flags=None):

    with tf.device('/gpu:0'):

        corner_valid_masks = tf.stack([tf.concat([tf.ones(NUM_WALL_CORNERS), tf.zeros(NUM_CORNERS - NUM_WALL_CORNERS)], axis=0),
                                       tf.ones(NUM_CORNERS),
                                       #tf.concat([tf.ones(NUM_WALL_CORNERS), tf.zeros(NUM_CORNERS - NUM_WALL_CORNERS)], axis=0),
                                       tf.zeros(NUM_CORNERS),
                                       tf.zeros(NUM_CORNERS),
                                       tf.ones(NUM_CORNERS)], axis=0)

        icon_valid_masks = tf.stack([tf.zeros(NUM_ICONS),
                                     tf.ones(NUM_ICONS),
                                     #tf.zeros(NUM_ICONS),
                                     tf.ones(NUM_ICONS),
                                     tf.ones(NUM_ICONS),
                                     tf.ones(NUM_ICONS)], axis=0)

        room_valid_masks = tf.stack([tf.zeros(NUM_ROOMS),
                                     tf.ones(NUM_ROOMS),
                                     #tf.zeros(NUM_ROOMS),
                                     tf.zeros(NUM_ROOMS),
                                     tf.ones(NUM_ROOMS),
                                     tf.ones(NUM_ROOMS)], axis=0)

        corner_valid_masks_bound = []
        for lossType in xrange(3):
            if lossType == 0:
                numChannels = NUM_WALL_CORNERS
            else:
                numChannels = 4
                pass
            if str(lossType) in options.loss:
                corner_valid_masks_bound.append(tf.ones(numChannels))
            else:
                corner_valid_masks_bound.append(tf.zeros(numChannels))
                pass
            continue
        corner_valid_masks_bound = tf.concat(corner_valid_masks_bound, axis=0)
        if '3' in options.loss:
            icon_valid_masks_bound = tf.ones(NUM_ICONS)
        else:
            icon_valid_masks_bound = tf.zeros(NUM_ICONS)
            pass
        if '4' in options.loss:
            room_valid_masks_bound = tf.ones(NUM_ROOMS)
        else:
            room_valid_masks_bound = tf.zeros(NUM_ROOMS)
            pass

        corner_valid_masks = tf.minimum(corner_valid_masks, tf.expand_dims(corner_valid_masks_bound, 0))
        icon_valid_masks = tf.minimum(icon_valid_masks, tf.expand_dims(icon_valid_masks_bound, 0))
        room_valid_masks = tf.minimum(room_valid_masks, tf.expand_dims(room_valid_masks_bound, 0))


        corners = gt_dict['corner']
        cornerSegmentation = tf.stack([tf.sparse_to_dense(tf.stack([corners[batchIndex, :, 1], corners[batchIndex, :, 0]], axis=1), (HEIGHT, WIDTH), corners[batchIndex, :, 2], validate_indices=False) for batchIndex in xrange(options.batchSize)], axis=0)
        cornerHeatmaps = tf.one_hot(cornerSegmentation, depth=NUM_CORNERS + 1, axis=-1)[:, :, :, 1:]

        # cornerHeatmaps = tf.one_hot(cornerSegmentation, depth=NUM_CORNERS, axis=-1)
        # kernel = tf.tile(tf.expand_dims(tf.constant(disk(11)), -1), [1, 1, NUM_CORNERS])
        # cornerHeatmaps = tf.nn.dilation2d(tf.expand_dims(cornerHeatmaps, 0), kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')[0]

        icon_gt = gt_dict['icon']
        room_gt = gt_dict['room']


        kernel_size = options.kernelSize

        neighbor_kernel_array = disk(kernel_size)
        neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
        neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
        cornerHeatmaps = tf.nn.depthwise_conv2d(cornerHeatmaps, tf.tile(neighbor_kernel, [1, 1, NUM_CORNERS, 1]), strides=[1, 1, 1, 1], padding='SAME')
        corner_gt = tf.cast(cornerHeatmaps > 0.5, tf.float32)
        gt_dict['corner_values'] = gt_dict['corner']
        gt_dict['corner'] = corner_gt

        if options.cornerLossType == 'softmax':
            corner_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits = pred_dict['corner'], labels = corner_gt, weights = tf.maximum(tf.cast(corner_gt > 0, tf.float32) * 100, 1)), axis=[0, 1, 2])
        elif options.cornerLossType == 'mse':
            #pred_corner = tf.sigmoid(pred_dict['corner'])
            #pred_corner = pred_corner * HEATMAP_SCALE
            kernel_size = 11
            #kernel_size = 5
            neighbor_kernel_array = gaussian(kernel_size)
            #neighbor_kernel_array = disk(kernel_size)
            neighbor_kernel_array /= neighbor_kernel_array.max()
            #print(neighbor_kernel_array)
            #exit(1)
            neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
            neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
            #heatmaps = 1 - tf.nn.max_pool(1 - heatmaps, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')
            corner_gt = tf.nn.depthwise_conv2d(cornerHeatmaps, tf.tile(neighbor_kernel, [1, 1, NUM_CORNERS, 1]), strides=[1, 1, 1, 1], padding='SAME')
            #corner_gt = tf.minimum(corner_gt * HEATMAP_SCALE, HEATMAP_SCALE)
            #print(pred_dict['corner'].shape, corner_gt.shape)
            #exit(1)
            corner_loss = tf.reduce_mean(tf.squared_difference(pred_dict['corner'], corner_gt), axis=[0, 1, 2])
            #pred_dict['corner'] /= HEATMAP_SCALE
        else:
            # kernel_size = 11
            # neighbor_kernel_array = disk(kernel_size)
            # neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
            # neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
            # heatmaps = tf.nn.depthwise_conv2d(heatmaps, tf.tile(neighbor_kernel, [1, 1, NUM_CORNERS, 1]), strides=[1, 1, 1, 1], padding='SAME')
            # heatmaps = tf.cast(heatmaps > 0.5, np.float32)
            #tune weight to 5

            corner_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = pred_dict['corner'], multi_class_labels = corner_gt, weights = tf.maximum(tf.cast(corner_gt > 0.5, tf.float32) * 5, 1), reduction=tf.losses.Reduction.NONE), axis=[0, 1, 2])
            pass


        #dataset_flag = flags[0][0]
        corner_loss = tf.reduce_mean(corner_loss * corner_valid_masks[dataset_flag])

        icon_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits = pred_dict['icon'], labels = icon_gt, weights = tf.maximum(tf.cast(icon_gt > 0, tf.float32) * options.iconPositiveWeight, 1), reduction=tf.losses.Reduction.NONE), axis=[0, 1, 2])
        icon_loss = tf.reduce_mean(icon_loss * icon_valid_masks[dataset_flag])
        room_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits = pred_dict['room'], labels = room_gt, reduction=tf.losses.Reduction.NONE), axis=[0, 1, 2])
        room_loss = tf.reduce_mean(room_loss * room_valid_masks[dataset_flag])
        #room_loss = tf.reduce_mean(tf.squared_difference(tf.squeeze(pred_room), tf.cast(segmentation_gt, tf.float32)))

        if options.branches == '4':
            corner_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(logits = pred_dict['corner'], multi_class_labels = corner_gt, weights = tf.maximum(tf.cast(corner_gt > 0.5, tf.float32) * 5, 1), reduction=tf.losses.Reduction.NONE), axis=[1, 2, 3])
            corner_loss = tf.reduce_sum(corner_loss * tf.cast(flags[:, 1], tf.float32))

            icon_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits = pred_dict['icon'], labels = icon_gt, weights = tf.maximum(tf.cast(icon_gt > 0, tf.float32) * 10, 1), reduction=tf.losses.Reduction.NONE), axis=[1, 2])
            icon_loss = tf.reduce_mean(icon_loss * tf.cast(flags[:, 1], tf.float32))

            room_loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits = pred_dict['room'], labels = room_gt, reduction=tf.losses.Reduction.NONE), axis=[1, 2])
            room_loss = tf.reduce_mean(room_loss * tf.cast(flags[:, 1], tf.float32))
            pass


        #corner_loss *= options.cornerLossWeight
        l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) * 0
        loss = corner_loss * options.cornerLossWeight + icon_loss * options.iconLossWeight + room_loss + l2_loss
        loss_list = []
        #losses.append(tf.cond(tf.equal(dataset_inp, 0), lambda: corner_loss, lambda: tf.constant(0.0)))
        #losses.append(tf.cond(tf.equal(dataset_inp, 1), lambda: corner_loss, lambda: tf.constant(0.0)))
        loss_list.append(corner_loss)
        loss_list.append(icon_loss)
        loss_list.append(room_loss)

        #debug_dict['room_mask'] = room_valid_masks[dataset_flag]
        pass

    return loss, loss_list


def train(options):
    if not os.path.exists(options.checkpoint_dir):
        os.system("mkdir -p %s"%options.checkpoint_dir)
        pass
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass
    if not os.path.exists(options.log_dir):
        os.system("mkdir -p %s"%options.log_dir)
        pass

    filenames_train = []
    if '0' in options.hybrid:
        filenames_train.append('data/Syn_train.tfrecords')
    if '1' in options.hybrid:
        filenames_train.append('data/Tango_train.tfrecords')
        pass
    if '2' in options.hybrid:
        filenames_train.append('data/ScanNet_train.tfrecords')
        pass
    if '3' in options.hybrid:
        filenames_train.append('data/Matterport_train.tfrecords')
        pass
    if '4' in options.hybrid:
        filenames_train.append('data/SUNCG_train.tfrecords')
        pass
    if options.slice:
        import RecordReaderSlice
        dataset_train = RecordReaderSlice.getDatasetTrain(filenames_train, options.augmentation, '4' in options.branches, options.batchSize)
    else:
        dataset_train = getDatasetTrain(filenames_train, options.augmentation, '4' in options.branches, options.batchSize)
    filenames_val = ['data/Tango_val.tfrecords']
    dataset_val = getDatasetVal(filenames_val, '', '4' in options.branches, options.batchSize)
    #dataset_val = dataset_train

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(handle, dataset_train.output_types, dataset_train.output_shapes)

    input_dict, gt_dict = iterator.get_next()

    iterator_train = dataset_train.make_one_shot_iterator()
    iterator_val = dataset_val.make_initializable_iterator()

    pred_dict, debug_dict = build_graph(options, input_dict)
    dataset_flag = input_dict['flags'][0, 0]
    if '4' in options.branches:
        loss, loss_list = build_loss(options, pred_dict, gt_dict, dataset_flag, debug_dict, input_dict['flags'])
    else:
        loss, loss_list = build_loss(options, pred_dict, gt_dict, dataset_flag, debug_dict)
        pass

    #training_flag = tf.placeholder(tf.bool, shape=[], name='training_flag')


    #with tf.variable_scope('statistics'):
    with tf.device('/cpu:0'):
        #batchno = tf.Variable(0, dtype=tf.int32)
        batchno = tf.Variable(0, dtype=tf.int32, trainable=False, name='batchno')
        batchnoinc = batchno.assign(batchno + 1)
        #optimizer = tf.train.AdamOptimizer(3e-3).minimize(loss, global_step=batchno, colocate_gradients_with_ops=True)
        pass

    optimizer = tf.train.AdamOptimizer(3e-3).minimize(loss, global_step=batchno)

    #tf.train.write_graph(tf.get_default_graph(), options.log_dir, 'train.pbtxt')
    writers_train = []
    writers_val = []
    for dataset in '01234':
        train_writer = tf.summary.FileWriter(options.log_dir + '/train_' + dataset)
        val_writer = tf.summary.FileWriter(options.log_dir + '/val_' + dataset)
        writers_train.append(train_writer)
        writers_val.append(val_writer)
        continue

    tf.summary.scalar('loss', loss)
    for index, l in enumerate(loss_list):
        tf.summary.scalar('loss_' + str(index), l)
        continue
    summary_op = tf.summary.merge_all()

    var_to_restore = [v for v in tf.global_variables()]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #config.log_device_placement=True
    saver=tf.train.Saver()

    threshold = np.ones((HEIGHT, WIDTH, 1)) * 0.5#HEATMAP_SCALE / 2

    profileTime = False
    if profileTime:
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        pass

    validation_losses = []
    with tf.Session(config=config) as sess:
        handle_train = sess.run(iterator_train.string_handle())
        handle_val = sess.run(iterator_val.string_handle())

        sess.run(tf.global_variables_initializer())
        tflearn.is_training(True)
        #if tf.train.checkpoint_exists("%s/checkpoint.ckpt"%(options.checkpoint_dir)):
        if options.restore == 1 and os.path.exists('%s/checkpoint.ckpt.index'%(options.checkpoint_dir)):
            #restore the same model from checkpoint
            print('restore')
            loader = tf.train.Saver(var_to_restore)
            if options.startIteration <= 0:
                loader.restore(sess, '%s/checkpoint.ckpt'%(options.checkpoint_dir))
            else:
                loader.restore(sess,"%s/checkpoint_%d.ckpt"%(options.checkpoint_dir, options.startIteration))
                pass
            bno = sess.run(batchno)
            print(bno)
        elif options.restore == 2 and os.path.exists('%s/checkpoint.ckpt.index'%(options.checkpoint_dir)):
            #restore the same model from checkpoint but reset batchno to 1
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess, '%s/checkpoint.ckpt'%(options.checkpoint_dir))
            sess.run(batchno.assign(1))
        elif options.restore == 3:
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess, '%s/checkpoint_%d.ckpt'%(options.checkpoint_dir, options.startIteration))
            bno = sess.run(batchno)
            print(bno)
        elif options.restore == 4:
            #fine-tune another model
            #var_to_restore = [v for v in var_to_restore if 'res4b22_relu_non_plane' not in v.name]
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess, '%s/checkpoint.ckpt'%(options.checkpoint_dir.replace('hybrid1', 'hybrid4')))
            sess.run(batchno.assign(1))
        elif options.restore == 5:
            var_to_restore = [v for v in var_to_restore if 'pred_' not in v.name]
            loader = tf.train.Saver(var_to_restore)
            loader.restore(sess, '%s/checkpoint.ckpt'%(options.checkpoint_dir))
            pass

        #if tf.train.checkpoint_exists("%s/%s.ckpt"%(dumpdir,keyname)):
        #saver.restore(sess,"%s/%s.ckpt"%(dumpdir,keyname))
        #pass

        MOVING_AVERAGE_DECAY = 0.99
        train_losses = [0., 0., 0.]
        train_acc = [1e-4, 1e-4, 1e-4]
        val_losses = [0., 0., 0.]
        val_acc = [1e-4, 1e-4, 1e-4]

        lastsave = time.time()
        bno = sess.run(batchno)

        #coord = tf.train.Coordinator()
        #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while bno < options.numIterations * (6.0 / options.batchSize):
            #while bno<64:
            try:
                if profileTime:
                    for iteration in xrange(5):
                        t0 = time.time()
                        # if options.slice:
                        #     mydebug = tf.get_collection("mydebug")[0]
                        #     _, total_loss, losses, summary_str, dataset, gt, pred, debug, mydebug_out = sess.run([optimizer, loss, loss_list, summary_op, dataset_flag, gt_dict, pred_dict, debug_dict, mydebug], feed_dict={handle: handle_train}, run_metadata=run_metadata, options=run_options)
                        # else:
                        _, total_loss, losses, summary_str, dataset, gt, pred, debug = sess.run([optimizer, loss, loss_list, summary_op, dataset_flag, gt_dict, pred_dict, debug_dict], feed_dict={handle: handle_train}, run_metadata=run_metadata, options=run_options)
                        print('time', time.time() - t0)
                        continue

                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('test/timeline.json', 'w') as f:
                        f.write(ctf)
                        pass
                    exit(1)
                    pass

                for iteration in xrange(500):
                    t0 = time.time()
                    if  iteration == 250 and options.visualize:
                        _, total_loss, losses, summary_str, dataset, gt, pred, debug = sess.run([optimizer, loss, loss_list, summary_op, dataset_flag, gt_dict, pred_dict, debug_dict], feed_dict={handle: handle_train})

                        visualizeBatch(options, 'train', pred, {'corner': gt['corner'], 'icon': gt['icon'], 'room': gt['room'], 'density': debug['x0_topdown'][:, :, :, -1]}, dataset)
                        # print(dataset)
                        # print(losses)
                        # print(gt['room'].dtype, gt['icon'].dtype)
                        # print(gt['room'][0].min(), gt['room'][0].max())
                        # print(pred['room'][0].min(), pred['room'][0].max())
                        # print(gt['icon'][0].min(), gt['icon'][0].max())
                        # print(pred['icon'][0].min(), pred['icon'][0].max())
                        # exit(1)
                    else:
                        _, total_loss, losses, summary_str, dataset = sess.run([optimizer, loss, loss_list, summary_op, dataset_flag], feed_dict={handle: handle_train})
                        pass

                    for lossIndex, value in enumerate(losses):
                        train_losses[lossIndex] = train_losses[lossIndex] * MOVING_AVERAGE_DECAY + value
                        train_acc[lossIndex] = train_acc[lossIndex] * MOVING_AVERAGE_DECAY + 1
                        continue
                    #print(bno + iteration, 't', train_losses[0] / train_acc[0], train_losses[1] / train_acc[1], train_losses[2] / train_acc[2], 'v', val_losses[0] / val_acc[0], val_losses[1] / val_acc[1], val_losses[2] / val_acc[2], time.time() - t0)
                    #print('dataset', dataset)
                    print('%d: t %02f %02f %02f, v %02f %02f %02f %02f' % (bno + iteration, train_losses[0] / train_acc[0], train_losses[1] / train_acc[1], train_losses[2] / train_acc[2], val_losses[0] / val_acc[0], val_losses[1] / val_acc[1], val_losses[2] / val_acc[2], time.time() - t0))
                    writers_train[dataset].add_summary(summary_str, bno + iteration)
                    continue
            except tf.errors.OutOfRangeError:
                print('Trained 1000 iterations')
                pass


            bno = sess.run(batchno)
            np.random.seed(bno)

            sess.run(iterator_val.initializer)
            try:
                #validation_loss = []
                for iteration in xrange(10):
                    if iteration == 0 and options.visualize:
                        total_loss, losses, summary_str, dataset, gt, pred, debug = sess.run([loss, loss_list, summary_op, dataset_flag, gt_dict, pred_dict, debug_dict], feed_dict={handle: handle_val})
                        visualizeBatch(options, 'val', pred, {'corner': gt['corner'], 'icon': gt['icon'], 'room': gt['room'], 'density': debug['x0_topdown'][:, :, :, -1]}, dataset)
                    else:
                        total_loss, losses, summary_str, dataset = sess.run([loss, loss_list, summary_op, dataset_flag], feed_dict={handle: handle_val})
                        pass
                    for lossIndex, value in enumerate(losses):
                        val_losses[lossIndex] = val_losses[lossIndex] * MOVING_AVERAGE_DECAY + value
                        val_acc[lossIndex] = val_acc[lossIndex] * MOVING_AVERAGE_DECAY + 1
                        continue
                    print('validation', 't', train_losses[0] / train_acc[0], train_losses[1] / train_acc[1], train_losses[2] / train_acc[2], 'v', val_losses[0] / val_acc[0], val_losses[1] / val_acc[1], val_losses[2] / val_acc[2])
                    writers_val[dataset].add_summary(summary_str, bno + iteration)
                    #validation_loss.append(total_loss)
                    continue
            except tf.errors.OutOfRangeError:
                print('Finish validation')
                pass
            pass

            # validation_losses.append(val_losses[0] / val_acc[0] + val_losses[1] / val_acc[1] + val_losses[2] / val_acc[2])
            # if len(validation_losses) >= 3 and validation_losses[-1] > validation_losses[-2] and validation_losses[-2] > validation_losses[-3]:
            #     print('validation losses', validation_losses)
            #     exit(1)
            #     pass

            print('save snapshot')
            saver.save(sess, "%s/checkpoint.ckpt"%(options.checkpoint_dir))
            if bno % 10000 == 0:
                saver.save(sess, "%s/checkpoint_%d.ckpt"%(options.checkpoint_dir, bno))
                pass

            print(bno, 't', train_losses[0] / train_acc[0], train_losses[1] / train_acc[1], train_losses[2] / train_acc[2], 'v', val_losses[0] / val_acc[0], val_losses[1] / val_acc[1], val_losses[2] / val_acc[2])
            continue
        pass
    return


def test(options):
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

    if options.useCache == 1 and os.path.exists(options.test_dir + '/network_numbers.npy'):
        numbers = np.load(options.test_dir + '/network_numbers.npy')[()]
        print([(k, v[0] / v[1], v[0] / v[2]) for k, v in numbers.iteritems()])
        #print(numbers)
        return numbers


    #print(options.checkpoint_dir)
    tf.reset_default_graph()

    filenames = []
    if '0' in options.dataset:
        filenames.append('data/Syn_val.tfrecords')
    if '1' in options.dataset:
        filenames.append('data/Tango_val.tfrecords')
        pass
    if '2' in options.dataset:
        filenames.append('data/ScanNet_val.tfrecords')
        pass
    if '3' in options.dataset:
        filenames.append('data/Matterport_val.tfrecords')
        pass
    if '4' in options.dataset:
        filenames.append('data/SUNCG_val.tfrecords')
        pass

    dataset = getDatasetVal(filenames, '', '4' in options.branches, options.batchSize)


    iterator = dataset.make_one_shot_iterator()
    input_dict, gt_dict = iterator.get_next()


    pred_dict, debug_dict = build_graph(options, input_dict)
    dataset_flag = input_dict['flags'][0, 0]
    flags = input_dict['flags'][:, 1]
    loss, loss_list = build_loss(options, pred_dict, gt_dict, dataset_flag, debug_dict)


    var_to_restore = [v for v in tf.global_variables()]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True
    #config.log_device_placement=True

    statisticsSum = {k: [0.0, 0.0, 0.0] for k in ['wall', 'door', 'icon', 'room']}

    numbers = {}

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        tflearn.is_training(False)
        #var_to_restore = [v for v in var_to_restore if 'pred_room' not in v.name]
        var_to_restore = [v for v in var_to_restore if 'is_training' not in v.name]
        loader = tf.train.Saver(var_to_restore)
        if options.startIteration <= 0:
            loader.restore(sess,"%s/checkpoint.ckpt"%(options.checkpoint_dir))
        else:
            loader.restore(sess,"%s/checkpoint_%d.ckpt"%(options.checkpoint_dir, options.startIteration))
            pass
        #if tf.train.checkpoint_exists("%s/%s.ckpt"%(dumpdir,keyname)):
        #saver.restore(sess,"%s/%s.ckpt"%(dumpdir,keyname))
        #pass

        MOVING_AVERAGE_DECAY = 1
        losses = [0., 0., 0.]
        acc = [1e-4, 1e-4, 1e-4]

        cornerCounters = {}
        for cornerType in CORNER_RANGES.keys():
            cornerCounters[cornerType] = np.zeros(3)
            pass

        globalCornerCounter = np.zeros(3)
        iconCounter = np.zeros(2)
        roomCounter = np.zeros(2)

        numImages = 0
        try:
            for iteration in xrange(options.numTestingImages):
                total_loss, losses, dataset, image_flags, gt, pred, debug, inp = sess.run([loss, loss_list, dataset_flag, flags, gt_dict, pred_dict, debug_dict, input_dict])

                for lossIndex, value in enumerate(losses):
                    losses[lossIndex] = losses[lossIndex] * MOVING_AVERAGE_DECAY + value
                    acc[lossIndex] = acc[lossIndex] * MOVING_AVERAGE_DECAY + 1
                    continue
                print('testing', losses[0] / acc[0], losses[1] / acc[1], losses[2] / acc[2])

                # print(iteration, inp['image_path'][0])
                # if iteration == 7:
                #     cv2.imwrite('test/density.png', drawDensityImage(getDensity(inp['points'][0], HEIGHT, WIDTH)))
                #     cv2.imwrite('test/density_indices.png', drawDensityImage(getDensityFromIndices(inp['point_indices'][0], HEIGHT, WIDTH)))
                #     exit(1)
                pred_c = sigmoid(pred['corner'])
                for batchIndex in xrange(options.batchSize):
                    if options.branches == '4' and image_flags[batchIndex] < 0.5:
                        continue
                    #print(inp['image_path'][batchIndex])

                    cornerHeatmapsGT = gt['corner'][batchIndex]
                    cornerHeatmapsPred = pred_c[batchIndex]
                    for cornerType, ranges in CORNER_RANGES.iteritems():
                        if options.cornerLossType == "mse":
                            cornersPred = extractCornersFromHeatmaps(cornerHeatmapsPred[:, :, ranges[0]:ranges[1]], heatmapThreshold=0.5, returnRanges=False)
                        else:
                            cornersPred = extractCornersFromHeatmaps(cornerHeatmapsPred[:, :, ranges[0]:ranges[1]], returnRanges=False)
                            pass
                        cornersGT = extractCornersFromHeatmaps(cornerHeatmapsGT[:, :, ranges[0]:ranges[1]], heatmapThreshold=0.5, returnRanges=False)

                        cornerStatistics = evaluateCorners(cornersPred, cornersGT)
                        cornerCounters[cornerType] += cornerStatistics
                        continue

                    globalCornersPred = []
                    for corners in cornersPred:
                        globalCornersPred += corners
                        continue
                    globalCornersGT = []
                    for corners in cornersGT:
                        globalCornersGT += corners
                        continue
                    globalCornerStatistics = evaluateCorners([globalCornersPred], [globalCornersGT])
                    globalCornerCounter += globalCornerStatistics

                    iconStatistics = evaluateSegmentation(np.argmax(pred['icon'][batchIndex], axis=-1), gt['icon'][batchIndex])
                    iconCounter += iconStatistics

                    roomStatistics = evaluateSegmentation(np.argmax(pred['room'][batchIndex], axis=-1), gt['room'][batchIndex])
                    roomCounter += roomStatistics

                    numImages += 1

                    continue

                statistics = visualizeBatch(options, 'test_' + str(iteration), pred, {'corner': gt['corner'], 'icon': gt['icon'], 'room': gt['room'], 'density': debug['x0_topdown'][:, :, :, -1], 'image_path': inp['image_path']}, dataset, savePredictions=False)

                for k, v in statistics.iteritems():
                    if k in statisticsSum:
                        for c in xrange(3):
                            statisticsSum[k][c] += v[c]
                            continue
                    continue
                continue
        except tf.errors.OutOfRangeError:
            print('Finish testing')
            pass

        for cornerType, cornerCounter in cornerCounters.iteritems():
            print(cornerType + ' corner recall and precision', cornerCounter[0] / cornerCounter[1], cornerCounter[0] / cornerCounter[2])
            continue
        #print('corner (type insensitive) recall and precision', globalCornerCounter[0] / globalCornerCounter[1], globalCornerCounter[0] / globalCornerCounter[2])
        print(('room accuracy and mean IOU', roomCounter[0] / numImages, roomCounter[1]/ numImages))
        print(('icon accuracy and mean IOU', iconCounter[0] / numImages, iconCounter[1]/ numImages))
        print('precision and recall', [(k, float(v[0]) / max(v[1], 1), float(v[0]) / max(v[2], 1)) for k, v in statisticsSum.iteritems()])
        #print(DATASETS[int(options.dataset[datasetIndex])], val_losses[0] / val_acc[0], val_losses[1] / val_acc[1], val_losses[2] / val_acc[2])

        # numbers['wall_corner'] = (cornerCounters['wall'][0] / cornerCounters['wall'][1], cornerCounters['wall'][0] / cornerCounters['wall'][2])
        # numbers['opening_corner'] = (cornerCounters['opening'][0] / cornerCounters['opening'][1], cornerCounters['opening'][0] / cornerCounters['opening'][2])
        # numbers['icon_corner'] = (cornerCounters['icon'][0] / cornerCounters['icon'][1], cornerCounters['icon'][0] / cornerCounters['icon'][2])
        # numbers['icon'] = (iconCounter[0] / numImages, iconCounter[1]/ numImages)
        # numbers['room'] = (roomCounter[0] / numImages, roomCounter[1]/ numImages)
        numbers['wall_corner'] = (cornerCounters['wall'][0], cornerCounters['wall'][1], cornerCounters['wall'][2])
        numbers['opening_corner'] = (cornerCounters['opening'][0], cornerCounters['opening'][1], cornerCounters['opening'][2])
        numbers['icon_corner'] = (cornerCounters['icon'][0], cornerCounters['icon'][1], cornerCounters['icon'][2])
        numbers['icon'] = (iconCounter[0], numImages, numImages)
        numbers['room'] = (roomCounter[0], numImages, numImages)
        pass


    if options.useCache != -1:
        np.save(options.test_dir + '/network_numbers.npy', numbers)
        pass
    print([(k, v[0] / v[1], v[0] / v[2]) for k, v in numbers.iteritems()])

    return numbers

def visualizeBatch(options, prefix, pred_dict, gt_dict, datasetFlag, savePredictions=False):
    # To update the modified time for options.test_dir
    if os.path.exists(options.test_dir + '/dummy'):
        #os.rmdir(options.test_dir + '/dummy')
        pass
    else:
        os.mkdir(options.test_dir + '/dummy')
        pass

    if savePredictions:
        print('save')
        np.save(options.test_dir + '/dummy/gt_dict.npy', gt_dict)
        np.save(options.test_dir + '/dummy/pred_dict.npy', pred_dict)
        exit(1)
        pass

    if options.cornerLossType != 'mse':
        threshold = np.ones((HEIGHT, WIDTH, 1)) * 0.5
    else:
        threshold = np.ones((HEIGHT, WIDTH, 1)) * 0.5# HEATMAP_SCALE / 2
        pass

    statisticsSum = {k: [0.0, 0.0, 0.0] for k in gt_dict}
    #print(pred_dict['corner'].max())
    pred_wc = pred_dict['corner'][:, :, :, :NUM_WALL_CORNERS]
    pred_oc = pred_dict['corner'][:, :, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4]
    pred_ic = pred_dict['corner'][:, :, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8]

    pred_wc = sigmoid(pred_wc)
    pred_oc = sigmoid(pred_oc)
    pred_ic = sigmoid(pred_ic)

    gt_wc = gt_dict['corner'][:, :, :, :NUM_WALL_CORNERS]
    gt_oc = gt_dict['corner'][:, :, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4]
    gt_ic = gt_dict['corner'][:, :, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8]

    for batchIndex in xrange(options.batchSize):
        #print('batch index', batchIndex)
        density = np.minimum(gt_dict['density'][batchIndex] * 255, 255).astype(np.uint8)
        density = np.stack([density, density, density], axis=2)
        #print('heatmap max value', pred_wc[batchIndex].max())
        if datasetFlag in [0, 1, 4]:
            cornerImage = drawSegmentationImage(np.concatenate([threshold, pred_wc[batchIndex]], axis=2), blackIndex=0)
            cornerImage[cornerImage == 0] = density[cornerImage == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_corner_pred.png', cornerImage)

            cornerImage = drawSegmentationImage(np.concatenate([threshold, gt_wc[batchIndex]], axis=2), blackIndex=0)
            cornerImage[cornerImage == 0] = density[cornerImage == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_corner_gt.png', cornerImage)
            pass


        if False:
            corner_heat = np.max(pred_wc[batchIndex], axis=-1)
            #print('corner_shape', corner_heat.shape)
            cmap = plt.get_cmap('jet')
            corner_rgba_img = cmap(corner_heat)
            corner_rgb_img = np.delete(corner_rgba_img, 3, 2)
            #print('rgb_out', corner_rgb_img.shape, corner_rgb_img.max(), corner_rgb_img.min())
            corner_rgb_img = (corner_rgb_img * 255).round().astype('uint8')
            #print('rgb_out', corner_rgb_img.shape, corner_rgb_img.max())
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_corner_heatmap.png', corner_rgb_img)
            pass

        if datasetFlag in [1, 4]:
            cornerImage = drawSegmentationImage(np.concatenate([threshold, pred_oc[batchIndex]], axis=2), blackIndex=0)
            cornerImage[cornerImage == 0] = density[cornerImage == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_opening_corner_pred.png', cornerImage)

            cornerImage = drawSegmentationImage(np.concatenate([threshold, pred_ic[batchIndex]], axis=2), blackIndex=0)
            cornerImage[cornerImage == 0] = density[cornerImage == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_icon_corner_pred.png', cornerImage)


            cornerImage = drawSegmentationImage(np.concatenate([threshold, gt_oc[batchIndex]], axis=2), blackIndex=0)
            cornerImage[cornerImage == 0] = density[cornerImage == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_opening_corner_gt.png', cornerImage)

            cornerImage = drawSegmentationImage(np.concatenate([threshold, gt_ic[batchIndex]], axis=2), blackIndex=0)
            cornerImage[cornerImage == 0] = density[cornerImage == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_icon_corner_gt.png', cornerImage)
            pass


        if datasetFlag in [1, 2, 3, 4]:
            icon_density = drawSegmentationImage(gt_dict['icon'][batchIndex], blackIndex=0)
            icon_density[icon_density == 0] = density[icon_density == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_icon_gt.png', icon_density)

            icon_density = drawSegmentationImage(pred_dict['icon'][batchIndex], blackIndex=0)
            icon_density[icon_density == 0] = density[icon_density == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_icon_pred.png', icon_density)
            pass

        if datasetFlag in [1, 3, 4]:
            room_density = drawSegmentationImage(gt_dict['room'][batchIndex], blackIndex=0)
            room_density[room_density == 0] = density[room_density == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_room_gt.png', room_density)

            room_density = drawSegmentationImage(pred_dict['room'][batchIndex], blackIndex=0)
            room_density[room_density == 0] = density[room_density == 0]
            cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_room_pred.png', room_density)
            pass


        if batchIndex == 0 and False:
            for c in xrange(22):
                cv2.imwrite(options.test_dir + '/mask_' + str(c) + '.png', cv2.dilate(drawMaskImage(corner_segmentation[batchIndex] == c), np.ones((3, 3)), 3))
                continue
            continue

        if batchIndex < options.visualizeReconstruction:
            #if batchIndex != 1:
            #continue
            print('reconstruct')
            try:
                from QP import reconstructFloorplan, findMatches
                gtHeatmaps = segmentation2Heatmaps(gt_c - 1, NUM_CORNERS)
                for heatmapIndex in xrange(gtHeatmaps.shape[-1]):
                    gtHeatmaps[:, :, heatmapIndex] = cv2.dilate(gtHeatmaps[:, :, heatmapIndex], np.ones((3, 3)), 7)
                    continue
                result_gt = reconstructFloorplan(gtHeatmaps[:, :, :NUM_WALL_CORNERS], gtHeatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4], gtHeatmaps[:, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8], segmentation2Heatmaps(gt_dict['icon'][batchIndex], NUM_ICONS), segmentation2Heatmaps(gt_dict['room'][batchIndex], NUM_ROOMS), density[:, :, 0], gt=True)
                resultImage = drawResultImage(WIDTH, HEIGHT, result_gt)
                cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_reconstruction_gt.png', resultImage)
                #if batchIndex == 1:
                #exit(1)
                pred_debug_dir = options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_debug'
                try:
                    os.mkdir(pred_debug_dir)
                    pass
                except OSError as e:
                    pass
                result_pred = reconstructFloorplan(pred_wc[batchIndex], pred_oc[batchIndex], pred_ic[batchIndex], sigmoid(pred_dict['icon'][batchIndex]), sigmoid(pred_dict['room'][batchIndex]), density[:, :, 0], gt_dict=gt_dict, gap=5, distanceThreshold=5, lengthThreshold=10, gt=False, debug_prefix=pred_debug_dir)
                #print(result_pred)
                if len(result_pred) == 0:
                    continue
                resultImage = drawResultImage(WIDTH, HEIGHT, result_pred)
                cv2.imwrite(options.test_dir + '/' + prefix + '_' + str(batchIndex) + '_reconstruction_pred.png', resultImage)

                print('find predictions among ground-truths')
                statistics = findMatches(result_pred, result_gt, distanceThreshold=10)
                print(statistics)
                print('statistics', [(k, float(v[0]) / max(v[1], 1), float(v[0]) / max(v[2], 1)) for k, v in statistics.iteritems()])

                for k, v in statistics.iteritems():
                    for c in xrange(3):
                        if c in statisticsSum:
                            statisticsSum[k][c] += v[c]
                        else:
                            print(c, 'not in', statisticsSum)
                        continue
                    continue

                #print('find ground-truths among predictions')
                #findMatches(result_gt, result_pred, distanceThreshold=10)
                #exit(1)
                pass
            except Exception as e:
                #traceback.print_tb(e)
                print('exception-----------: ', e)
                #raise e
        continue
    #exit(1)
    return statisticsSum




def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Planenet')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    #task: [train, test, predict]
    parser.add_argument('--task', dest='task',
                        help='task type: [train, test, predict]',
                        default='train', type=str)
    parser.add_argument('--restore', dest='restore',
                        help='how to restore the model',
                        default=1, type=int)
    parser.add_argument('--batchSize', dest='batchSize',
                        help='batch size',
                        default=6, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        help='dataset name for test/predict',
                        default='1', type=str)
    parser.add_argument('--slice', dest='slice', help='whether or not to use the slice version.',
                        action='store_true')
    parser.add_argument('--numTestingImages', dest='numTestingImages',
                        help='the number of images to test/predict',
                        default=20, type=int)
    parser.add_argument('--fineTuningCheckpoint', dest='fineTuningCheckpoint',
                        help='specify the model for fine-tuning',
                        default='checkpoint/floornet_hybrid4_branch0123_wsf', type=str)
    parser.add_argument('--suffix', dest='suffix',
                        help='add a suffix to keyname to distinguish experiments',
                        default='', type=str)
    parser.add_argument('--l2Weight', dest='l2Weight',
                        help='L2 regulation weight',
                        default=5e-4, type=float)
    parser.add_argument('--LR', dest='LR',
                        help='learning rate',
                        default=3e-5, type=float)
    parser.add_argument('--hybrid', dest='hybrid',
                        help='hybrid training',
                        default='1', type=str)
    parser.add_argument('--branches', help='active branches of the network: 0: PointNet, 1: top-down, 2: bottom-up, 3: PointNet segmentation, 4: Image Features, 5: Image Features with Joint training, 6: Additional Layers Before Pred (0, 01, 012, 0123, 01234, 1, 02*, 013)',
                        default='0123', type=str)
    #parser.add_argument('--batch_norm', help='add batch normalization to network', action='store_true')

    parser.add_argument('--cornerLossType', dest='cornerLossType',
                        help='corner loss type',
                        default='sigmoid', type=str)
    parser.add_argument('--loss',
                        help='loss type needed. [wall corner loss, door corner loss, icon corner loss, icon segmentation, room segmentation]',
                        default='01234', type=str)
    parser.add_argument('--cornerLossWeight', dest='cornerLossWeight',
                        help='corner loss weight',
                        default=10, type=float)
    parser.add_argument('--augmentation', dest='augmentation',
                        help='augmentation (wsfd)',
                        default='wsf', type=str)
    parser.add_argument('--numPoints', dest='numPoints',
                        help='number of points',
                        default=50000, type=int)
    parser.add_argument('--numInputChannels', dest='numInputChannels',
                        help='number of input channels',
                        default=7, type=int)
    parser.add_argument('--sumScale', dest='sumScale',
                        help='avoid segment sum results to be too large',
                        default=10, type=int)
    parser.add_argument('--visualizeReconstruction', dest='visualizeReconstruction',
                        help='whether to visualize flooplan reconstruction or not',
                        default=0, type=int)
    parser.add_argument('--numFinalChannels', dest='numFinalChannels', help='the number of final channels', default=256, type=int)
    parser.add_argument('--numIterations', dest='numIterations', help='the number of iterations', default=10000, type=int)
    parser.add_argument('--startIteration', dest='startIteration', help='the index of iteration to start', default=0, type=int)
    parser.add_argument('--useCache', dest='useCache',
                        help='whether to cache or not',
                        default=1, type=int)
    parser.add_argument('--debug', dest='debug',
                        help='debug index',
                        default=-1, type=int)
    parser.add_argument('--outputLayers', dest='outputLayers',
                        help='output layers',
                        default='two', type=str)
    parser.add_argument('--kernelSize', dest='kernelSize',
                        help='corner kernel size',
                        default=11, type=int)
    parser.add_argument('--iconLossWeight', dest='iconLossWeight',
                        help='icon loss weight',
                        default=1, type=float)
    parser.add_argument('--poolingTypes', dest='poolingTypes',
                        help='pooling types',
                        default='sssmm', type=str)
    parser.add_argument('--visualize', dest='visualize',
                        help='visualize during training',
                        action='store_false')
    parser.add_argument('--iconPositiveWeight', dest='iconPositiveWeight',
                        help='icon positive weight',
                        default=10, type=int)
    parser.add_argument('--prefix', dest='prefix',
                        help='prefix',
                        default='floornet', type=str)
    parser.add_argument('--drawFinal', dest='drawFinal',
                        help='draw final',
                        action='store_false')
    parser.add_argument('--separateIconLoss', dest='separateIconLoss',
                        help='separate loss for icon',
                        action='store_false')
    parser.add_argument('--evaluateImage', dest='evaluateImage',
                        help='evaluate image',
                        action='store_true')


    args = parser.parse_args()

    #args.keyname = os.path.basename(__file__).rstrip('.py')
    #args.keyname = args.keyname.replace('train_', '')
    #layers where deep supervision happens
    addArgs(args)
    return args


def addArgs(args):
    #args.keyname = 'floornet'
    args.keyname = args.prefix

    args.keyname += '_hybrid' + args.hybrid
    args.keyname += '_branch' + args.branches
    if args.cornerLossType != 'sigmoid':
        args.keyname += '_' + args.cornerLossType
        pass

    if args.augmentation != '':
        args.keyname += '_' + args.augmentation
        pass

    if args.numPoints != 50000:
        args.keyname += '_' + str(args.numPoints)
        pass
    if args.loss != '01234':
        args.keyname += '_loss' + str(args.loss)
        pass

    if args.numFinalChannels != 256:
        args.keyname += '_' + str(args.numFinalChannels)
        NUM_CHANNELS[-1] = args.numFinalChannels
        pass
    # if args.numIterations != 10000:
    #     args.keyname += '_' + str(args.numIterations)
    #     pass
    if args.outputLayers != 'two':
        args.keyname += '_' + str(args.outputLayers)
        pass

    if args.kernelSize != 11:
        args.keyname += '_' + str(args.kernelSize)
        pass

    if args.iconLossWeight != 1:
        args.keyname += '_' + str(args.iconLossWeight)
        pass

    if args.sumScale != 10:
        args.keyname += '_' + str(args.sumScale)
        pass

    if args.poolingTypes != 'sssmm':
        args.keyname += '_' + str(args.poolingTypes)
        pass

    if args.iconPositiveWeight != 10:
        args.keyname += '_' + str(args.iconPositiveWeight)
        pass
    if args.slice:
        args.keyname += '_slice'
        pass
    # if args.batch_norm:
    #     args.keyname += '_batchnorm'
    args.checkpoint_dir = 'checkpoint/' + args.keyname
    args.log_dir = 'log/' + args.keyname
    args.test_dir = 'test/' + args.keyname
    if args.task == 'test':
        args.test_dir += '/dataset_' + args.dataset
        args.batchSize = 1
        pass
    if args.task == 'predict':
        args.predict_dir = 'predict/' + args.keyname + '_' + args.dataset
        pass
    args.dump_dir = 'dump/' + args.keyname


if __name__=='__main__':
    args = parse_args()

    print('keyname=%s task=%s started'%(args.keyname, args.task))
    if args.task == 'train':
        train(args)
    elif args.task == 'test':
        args.batchSize = 1
        if args.dataset == '':
            args.dataset = args.hybrid
            pass
        if '1' in args.dataset:
            args.dataset = '1'
            pass
        test(args)
    elif args.task == 'check':
        #args.batchSize = 1
        args.test_dir += '/dataset_' + args.dataset
        evaluateBatch(args)
    elif args.task == 'evaluate':
        from evaluate import *
        args.batchSize = 1
        if args.loss == '5':
            hybrid = args.hybrid
            for loss in '01234':
                args.loss = loss
                if hybrid == '14':
                    if '0' in loss and '1' in loss:
                        args.hybrid = '14'
                    else:
                        args.hybrid = '1'
                        pass
                addArgs(args)
                args.test_dir += '/dataset_' + args.dataset
                evaluate(args)
                continue
            args.hybrid = hybrid
            args.useCache = 1
            args.loss = '5'
            args.test_dir += '/dataset_' + args.dataset
            addArgs(args)
            evaluateBatch(args)
        else:
            args.test_dir += '/dataset_' + args.dataset
            evaluate(args)
            pass
    elif args.task == 'tango':
        from evaluate_tango import *
        args.batchSize = 1
        args.test_dir += '/dataset_' + args.dataset + '_tango'
        evaluateTango(args)
    elif args.task == 'predict':
        predict(args)
    else:
        assert False, 'format wrong'
        pass
