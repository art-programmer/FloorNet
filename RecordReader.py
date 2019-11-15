import tensorflow as tf
import numpy as np
import PIL.Image as Image
import sys
import os
import functools
from floorplan_utils import *
from augmentation_tf import *
from utils import *
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NUM_THREADS = 8

def parse_fn(example, augmentation, readImageFeatures = False, convertCorners=False, kernelSize=11):
    if readImageFeatures:
        features = tf.parse_single_example(
            example,
            # Defaults are not specified since both keys are required.
            features={
                'image_path': tf.FixedLenFeature([], tf.string),
                'points': tf.FixedLenFeature([NUM_POINTS * (NUM_INPUT_CHANNELS - 1)], tf.float32),
                'point_indices': tf.FixedLenFeature([NUM_POINTS], tf.int64),
                'corner': tf.FixedLenFeature([MAX_NUM_CORNERS * 3], tf.int64),
                'num_corners': tf.FixedLenFeature([], tf.int64),
                'icon': tf.FixedLenFeature([], tf.string),
                'room': tf.FixedLenFeature([], tf.string),
                'image': tf.FixedLenFeature(sum([size * size * numChannels for size, numChannels in zip(SIZES, NUM_CHANNELS)[1:]]), tf.float32),
                'flags': tf.FixedLenFeature([2], tf.int64),
            })
    else:
        features = tf.parse_single_example(
            example,
            # Defaults are not specified since both keys are required.
            features={
                'image_path': tf.FixedLenFeature([], tf.string),
                'points': tf.FixedLenFeature([NUM_POINTS * (NUM_INPUT_CHANNELS - 1)], tf.float32),
                'point_indices': tf.FixedLenFeature([NUM_POINTS], tf.int64),
                'corner': tf.FixedLenFeature([MAX_NUM_CORNERS * 3], tf.int64),
                'num_corners': tf.FixedLenFeature([], tf.int64),
                'icon': tf.FixedLenFeature([], tf.string),
                'room': tf.FixedLenFeature([], tf.string),
                'flags': tf.FixedLenFeature([2], tf.int64),
            })
        pass

    points = tf.reshape(features['points'], (NUM_POINTS, NUM_INPUT_CHANNELS - 1))
    point_indices = tf.cast(features['point_indices'], tf.int32)
    #point_indices = features['point_indices']

    corners = tf.cast(tf.reshape(features['corner'], [MAX_NUM_CORNERS, 3]), tf.int32)
    numCorners = features['num_corners']
    corners = corners[:numCorners]
    iconSegmentation = tf.reshape(tf.decode_raw(features['icon'], tf.uint8), (HEIGHT, WIDTH, 1))
    roomSegmentation = tf.reshape(tf.decode_raw(features['room'], tf.uint8), (HEIGHT, WIDTH, 1))
    heatmaps = tf.stack([iconSegmentation, roomSegmentation], axis=0)

    if readImageFeatures:
        imageFeature = features['image']
        imageFeatures = {}
        offset = 0
        for index, (size, numChannels) in enumerate(zip(SIZES, NUM_CHANNELS)[1:]):
            imageFeatures[index] = tf.reshape(imageFeature[offset:offset + size * size * numChannels], (size, size, numChannels))
            offset += size * size * numChannels
            continue
    else:
        imageFeatures = {}
        pass

    flags = features['flags']
    if 'w' in augmentation:
        point_indices, corners, heatmaps = tf.cond(tf.logical_or(tf.equal(flags[0], 0), tf.equal(flags[0], 4)), lambda: augmentWarping(point_indices, corners, heatmaps, gridStride=32, randomScale=2), lambda: (point_indices, corners, heatmaps))
        #point_indices, corners, heatmaps = augmentWarping(point_indices, corners, heatmaps, gridStride=32, randomScale=4)
        pass
    if 's' in augmentation:
        points, point_indices, corners, heatmaps, imageFeatures = augmentScaling(points, point_indices, corners, heatmaps, imageFeatures)
        pass
    if 'f' in augmentation:
        points, point_indices, corners, heatmaps, imageFeatures = augmentFlipping(points, point_indices, corners, heatmaps, imageFeatures)
        pass

    iconSegmentation = tf.cast(tf.squeeze(heatmaps[0]), tf.int32)
    roomSegmentation = tf.cast(tf.squeeze(heatmaps[1]), tf.int32)

    roomSegmentation = tf.minimum(roomSegmentation, NUM_ROOMS - 1)

    # point_indices_stack = getCoarseIndicesMaps(point_indices, WIDTH, HEIGHT, 0)

    corners = tf.reshape(tf.concat([corners, tf.zeros((MAX_NUM_CORNERS - tf.shape(corners)[0], 3), dtype=tf.int32)], axis=0), (MAX_NUM_CORNERS, 3))
    if convertCorners:
        cornerSegmentation = tf.stack([tf.sparse_to_dense(tf.stack([corners[:, 1], corners[:, 0]], axis=1), (HEIGHT, WIDTH), corners[:, 2], validate_indices=False)], axis=0)
        cornerHeatmaps = tf.one_hot(cornerSegmentation, depth=NUM_CORNERS + 1, axis=-1)[:, :, :, 1:]
        #kernel_size = kernelSize
        #neighbor_kernel_array = disk(kernel_size)
        #neighbor_kernel = tf.constant(neighbor_kernel_array.reshape(-1), shape=neighbor_kernel_array.shape, dtype=tf.float32)
        #neighbor_kernel = tf.reshape(neighbor_kernel, [kernel_size, kernel_size, 1, 1])
        #cornerHeatmaps = tf.nn.depthwise_conv2d(cornerHeatmaps, tf.tile(neighbor_kernel, [1, 1, NUM_CORNERS, 1]), strides=[1, 1, 1, 1], padding='SAME')
        corners = tf.cast(cornerHeatmaps > 0.5, tf.float32)

    # cornerSegmentation = tf.sparse_to_dense(tf.stack([corners[:, 1], corners[:, 0]], axis=1), (HEIGHT, WIDTH), corners[:, 2], validate_indices=False)
    # cornerHeatmaps = tf.one_hot(cornerSegmentation, depth=NUM_CORNERS, axis=-1)
    # kernel = tf.tile(tf.expand_dims(tf.constant(disk(11)), -1), [1, 1, NUM_CORNERS])
    # cornerHeatmaps = tf.nn.dilation2d(tf.expand_dims(cornerHeatmaps, 0), kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')[0]

    imagePath = features['image_path']

    points = tf.concat([points, tf.ones((NUM_POINTS, 1))], axis=1)

    if readImageFeatures:
        input_dict = {'points': points, 'point_indices': point_indices, 'image_features': imageFeatures, 'image_path': imagePath, 'flags': flags}
    else:
        input_dict = {'points': points, 'point_indices': point_indices, 'image_path': imagePath, 'flags': flags}
        pass
    gt_dict = {'corner': corners, 'icon': iconSegmentation, 'room': roomSegmentation, 'num_corners': numCorners}
    return input_dict, gt_dict

# class RecordReader():
#     def __init__(self):
#         return

#     def getBatch(self, filename_queues, batchSize, augmentation, min_after_dequeue = 1000, random=True, getLocal=False, getSegmentation=False, test=True):
#         reader = tf.TFRecordReader()
#         queueIndex = tf.cast(tf.random_uniform([len(filename_queues)], maxval=5), tf.int32)[0]
#         #filename_queue = tf.cond(tf.equal(queueIndex, 0), lambda: filename_queues[0], lambda: tf.cond(tf.equal(queueIndex, 1), lambda: filename_queues[1], lambda: tf.cond(tf.equal(queueIndex, 2), lambda: filename_queues[2], lambda: tf.cond(tf.equal(queueIndex, 3), lambda: filename_queues[3], lambda: filename_queues[4]))))

#         _, serialized_example = reader.read(filename_queues[queueIndex])

#         features = tf.parse_single_example(
#             serialized_example,
#             # Defaults are not specified since both keys are required.
#             features={
#                 'image_path': tf.FixedLenFeature([], tf.string),
#                 'points': tf.FixedLenFeature([NUM_POINTS * 7], tf.float32),
#                 'point_indices': tf.FixedLenFeature([NUM_POINTS], tf.int32),
#                 'corner': tf.FixedLenFeature([MAX_NUM_CORNERS * 3], tf.int32),
#                 'num_corners': tf.FixedLenFeature([], tf.int32),
#                 'icon': tf.FixedLenFeature([], tf.string),
#                 'room': tf.FixedLenFeature([], tf.string),
#                 'image': tf.FixedLenFeature(sum([size * size * numChannels for size, numChannels in zip(SIZES, NUM_CHANNELS)[1:]]), tf.float32),
#                 'flags': tf.FixedLenFeature([2], tf.int32),
#             })


#         points = tf.reshape(features['points'], (NUM_POINTS, NUM_INPUT_CHANNELS))
#         point_indices = tf.reshape(features['point_indices'], (NUM_POINTS))
#         corners = tf.reshape(features['corner'], [MAX_NUM_CORNERS, 3])
#         numCorners = tf.cast(features['num_corners'], tf.int32)
#         corners = corners[:numCorners]
#         iconSegmentation = tf.reshape(tf.decode_raw(features['icon'], tf.uint8), (HEIGHT, WIDTH, 1))
#         roomSegmentation = tf.reshape(tf.decode_raw(features['room'], tf.uint8), (HEIGHT, WIDTH, 1))
#         heatmaps = tf.stack([icon, room], axis=0)

#         flags = features['flags']

#         if 'w' in augmentation:
#             point_indices, corners, heatmaps = tf.cond(flags[0] == 0 or flags[0] == 4, lambda: augmentWarping(point_indices, corners, heatmaps, stride=32, randomScale=4), lambda: pointcloudIndices, corners, heatmaps)
#             pass
#         if 's' in augmentation:
#             points, point_indices, corners, heatmaps = augmentScaling(points, point_indices, corners, heatmaps)
#             pass
#         if 'f' in augmentation:
#             points, point_indices, corners, heatmaps = augmentFlipping(points, point_indices, corners, heatmaps)
#             pass

#         iconSegmentation = tf.squeeze(heatmaps[0])
#         roomSegmentation = tf.squeeze(heatmaps[1])

#         point_indices_stack = getCoarseIndicesMaps(point_indices, WIDTH, HEIGHT, 0)

#         cornerSegmentation = tf.sparse_to_dense(tf.stack([corners[:, 1], corners[:, 0]], axis=1), (HEIGHT, WIDTH), corners[:, 2])
#         cornerHeatmaps = tf.one_hot(cornerSegmentation, depth=NUM_CORNERS, axis=-1)

#         kernel = tf.tile(tf.expand_dims(tf.constant(disk(11)), -1), [1, 1, NUM_CORNERS])
#         cornerHeatmaps = nn.diltion2d(cornerHeatmaps, kernel, [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')

#         imagePath = features['image_path']

#         imageFeatures = []
#         imageFeature = tf.reduce_sum(tf.reshape(features['image'], [2, -1]), axis=0)
#         offset = 0
#         for size, numChannels in zip(SIZES, NUM_CHANNELS)[1:]:
#             imageFeatures.append(tf.reshape(imageFeature[offset:offset + size * size * numChannels], (size[1], size, numChannels)))
#             offset += size * size * numChannels
#             continue

#         if random:
#             points_inp, point_indices_inp, corner_gt, icon_gt, room_gt, image_features_inp, image_path_inp, flags_inp = tf.train.shuffle_batch([points, point_indices_stack, cornerHeatmaps, iconSegmentation, roomSegmentation, imageFeatures, imagePath, flags], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=NUM_THREADS, min_after_dequeue=min_after_dequeue)
#         else:
#             points_inp, point_indices_inp, corner_gt, icon_gt, room_gt, image_features_inp, image_path_inp, flags_inp = tf.train.batch([points, point_indices_stack, cornerHeatmaps, iconSegmentation, roomSegmentation, imageFeatures, imagePath, flags], batch_size=batchSize, capacity=min_after_dequeue + (NUM_THREADS + 2) * batchSize, num_threads=1)
#             pass

#         input_dict = {'points': points_inp, 'point_indices': point_indices_inp, 'image_features': image_features_inp, 'image_path': image_path_inp, 'flags': flags_inp}
#         gt_dict = {'corner': corner_gt, 'icon': icon_gt, 'room': room_gt}
#         return input_dict, gt_dict


def getDatasetTrain(filenames, augmentation, readImageFeatures, batchSize):
    if len(filenames) == 1:
        return tf.data.TFRecordDataset(filenames[0]).repeat().map(functools.partial(parse_fn, augmentation=augmentation, readImageFeatures=readImageFeatures), num_parallel_calls=NUM_THREADS).batch(batchSize).prefetch(1)
    else:
        return tf.data.Dataset.from_tensor_slices(filenames).interleave(lambda x: tf.data.TFRecordDataset(x).repeat().map(functools.partial(parse_fn, augmentation=augmentation), num_parallel_calls=NUM_THREADS), cycle_length=len(filenames), block_length=batchSize).batch(batchSize).prefetch(100 / len(filenames)).shuffle(100 / len(filenames))


def getDatasetVal(filenames, augmentation, readImageFeatures, batchSize):
    #return tf.data.Dataset.from_tensor_slices(filenames).interleave(lambda x: tf.data.TFRecordDataset(x).map(functools.partial(parse_fn, augmentation=augmentation, readImageFeatures=readImageFeatures), num_parallel_calls=NUM_THREADS), cycle_length=len(filenames), block_length=batchSize).apply(tf.contrib.data.batch_and_drop_remainder(batchSize)).prefetch(1)
    if len(filenames) == 1:
        return tf.data.TFRecordDataset(filenames[0]).repeat().map(functools.partial(parse_fn, augmentation=augmentation, readImageFeatures=readImageFeatures), num_parallel_calls=NUM_THREADS).batch(batchSize).prefetch(1)
    else:
        return tf.data.Dataset.from_tensor_slices(filenames).interleave(lambda x: tf.data.TFRecordDataset(x).map(functools.partial(parse_fn, augmentation=augmentation, readImageFeatures=readImageFeatures), num_parallel_calls=NUM_THREADS), cycle_length=len(filenames), block_length=batchSize).apply(tf.contrib.data.batch_and_drop_remainder(batchSize)).prefetch(1)
