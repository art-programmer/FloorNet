import numpy as np
from utils import *
import tensorflow as tf
from floorplan_utils import *


def warpIndices(xs, ys, gridStride, gridWidth, gridHeight, width, height, gridXsTarget, gridYsTarget):
    numPoints = xs.shape[0]

    minXs = xs / gridStride
    minYs = ys / gridStride
    maxXs = xs / gridStride + 1
    maxYs = ys / gridStride + 1

    topLeft = tf.expand_dims(minYs * gridWidth + minXs, -1)
    topRight = tf.expand_dims(minYs * gridWidth + maxXs, -1)
    bottomLeft = tf.expand_dims(maxYs * gridWidth + minXs, -1)
    bottomRight = tf.expand_dims(maxYs * gridWidth + maxXs, -1)

    topLeftXsTarget = tf.gather_nd(gridXsTarget, topLeft)
    topLeftYsTarget = tf.gather_nd(gridYsTarget, topLeft)
    topRightXsTarget = tf.gather_nd(gridXsTarget, topRight)
    topRightYsTarget = tf.gather_nd(gridYsTarget, topRight)
    bottomLeftXsTarget = tf.gather_nd(gridXsTarget, bottomLeft)
    bottomLeftYsTarget = tf.gather_nd(gridYsTarget, bottomLeft)
    bottomRightXsTarget = tf.gather_nd(gridXsTarget, bottomRight)
    bottomRightYsTarget = tf.gather_nd(gridYsTarget, bottomRight)

    ratioX = tf.cast(xs - minXs * gridStride, tf.float32) / float(gridStride)
    ratioY = tf.cast(ys - minYs * gridStride, tf.float32) / float(gridStride)
    topLeftRatio = (1 - ratioX) * (1 - ratioY)
    topRightRatio = ratioX * (1 - ratioY)
    bottomLeftRatio = (1 - ratioX) * ratioY
    bottomRightRatio = ratioX * ratioY

    xsTarget = topLeftXsTarget * topLeftRatio + topRightXsTarget * topRightRatio + bottomLeftXsTarget * bottomLeftRatio + bottomRightXsTarget * bottomRightRatio
    ysTarget = topLeftYsTarget * topLeftRatio + topRightYsTarget * topRightRatio + bottomLeftYsTarget * bottomLeftRatio + bottomRightYsTarget * bottomRightRatio

    xsTarget = tf.clip_by_value(tf.cast(tf.round(xsTarget), tf.int32), 0, width - 1)
    ysTarget = tf.clip_by_value(tf.cast(tf.round(ysTarget), tf.int32), 0, height - 1)
    return xsTarget, ysTarget

def scaleIndices(xs, ys, min_x, min_y, max_x, max_y, width, height):
    xsTarget = (tf.cast(xs, tf.float32) - min_x) / (max_x - min_x + 1) * width
    ysTarget = (tf.cast(ys, tf.float32) - min_y) / (max_y - min_y + 1) * height
    xsTarget = tf.clip_by_value(tf.cast(tf.round(xsTarget), tf.int32), 0, width - 1)
    ysTarget = tf.clip_by_value(tf.cast(tf.round(ysTarget), tf.int32), 0, height - 1)
    return xsTarget, ysTarget

#Get coarse indices maps from 256x256 indices map
def getCoarseIndicesMaps(indicesMap, width=256, height=256, batchIndex=0):
    indicesMaps = []
    for strideIndex in xrange(6):
        stride = pow(2, strideIndex)
        if strideIndex == 0:
            indicesMaps.append(indicesMap + batchIndex * width * height)
        else:
            indicesMaps.append(indicesMap / (width * stride) * (width / stride) + indicesMap % width / stride + batchIndex * width / stride * height / stride)
        #print(indicesMaps)
        continue
    indicesMaps = tf.stack(indicesMaps, axis=0)
    return indicesMaps

#Get coarse indices maps from 256x256 indices map
def getCoarseIndicesMapsBatch(indicesMap, width=256, height=256):
    indicesMaps = []
    for strideIndex in xrange(6):
        stride = pow(2, strideIndex)
        if strideIndex == 0:
            indicesMaps.append(indicesMap)
        else:
            indicesMaps.append(indicesMap / (width * stride) * (width / stride) + indicesMap % width / stride)
        #print(indicesMaps)
        continue
    indicesMaps = tf.stack(indicesMaps, axis=0)
    return indicesMaps


def augmentWarping(pointcloudIndices, corners, heatmaps, gridStride=16, randomScale=4):
    width = WIDTH
    height = HEIGHT
    gridWidth = int(width / gridStride + 1)
    gridHeight = int(height / gridStride + 1)

    gridXs = tf.reshape(tf.tile(tf.expand_dims(tf.range(gridWidth) * gridStride, 0), [gridHeight, 1]), [-1])
    gridYs= tf.reshape(tf.tile(tf.expand_dims(tf.range(gridHeight) * gridStride, -1), [1, gridWidth]), [-1])

    gridXsTarget = tf.cast(gridXs, tf.float32) + tf.random_normal(stddev=randomScale, shape=[gridHeight * gridWidth])
    gridYsTarget = tf.cast(gridYs, tf.float32) + tf.random_normal(stddev=randomScale, shape=[gridHeight * gridWidth])

    xsTarget, ysTarget = warpIndices(pointcloudIndices % width, pointcloudIndices / width, gridStride, gridWidth, gridHeight, width, height, gridXsTarget, gridYsTarget)

    newPointcloudIndices = tf.clip_by_value(ysTarget, 0, height - 1) * width + tf.clip_by_value(xsTarget, 0, width - 1)

    xsTarget, ysTarget = warpIndices(corners[:, 0], corners[:, 1], gridStride, gridWidth, gridHeight, width, height, gridXsTarget, gridYsTarget)
    newCorners = tf.stack([xsTarget, ysTarget, corners[:, 2]], axis=1)

    return newPointcloudIndices, newCorners, heatmaps

def augmentScaling(pointcloud, pointcloudIndices, corners, heatmaps, imageFeatures):
    width = WIDTH
    height = HEIGHT
    xs = pointcloudIndices % width
    ys = pointcloudIndices / width

    imageSize = tf.constant((height, width), dtype=np.float32)
    #randomScale = pow(2.0, tf.random.uniform([1]) - 1)
    randomScale = tf.random_uniform(shape=[1], minval=0.5, maxval=1.5)[0]
    xsTarget = tf.clip_by_value(tf.cast(tf.round((tf.cast(xs, tf.float32) - width / 2) * randomScale + width / 2), tf.int32), 0, width - 1)
    ysTarget = tf.clip_by_value(tf.cast(tf.round((tf.cast(ys, tf.float32) - height / 2) * randomScale + height / 2), tf.int32), 0, height - 1)


    newPointcloudIndices = ysTarget * width + xsTarget

    pointcloud = (pointcloud - 0.5) * randomScale + 0.5

    newHeatmaps = tf.image.resize_nearest_neighbor(heatmaps, size = tf.cast(tf.round(imageSize * randomScale), tf.int32))
    newHeatmaps = tf.image.resize_image_with_crop_or_pad(newHeatmaps, height, width)

    xsTarget = tf.cast(tf.round((tf.cast(corners[:, 0], tf.float32) - width / 2) * randomScale + width / 2), tf.int32)
    ysTarget = tf.cast(tf.round((tf.cast(corners[:, 1], tf.float32) - height / 2) * randomScale + height / 2), tf.int32)
    newCorners = tf.stack([xsTarget, ysTarget, corners[:, 2]], axis=1)
    validMask = tf.logical_and(tf.logical_and(tf.greater_equal(newCorners[:, 0], 0), tf.greater_equal(newCorners[:, 1], 0)), tf.logical_and(tf.less(newCorners[:, 0], WIDTH), tf.less(newCorners[:, 1], HEIGHT)))
    newCorners = tf.boolean_mask(newCorners, validMask)

    for index, (featureSize, numChannels) in enumerate(zip(SIZES, NUM_CHANNELS)[1:]):
        if index in imageFeatures:
            imageFeatures[index] = tf.image.resize_image_with_crop_or_pad(tf.image.resize_nearest_neighbor(tf.expand_dims(imageFeatures[index], 0), size = tf.cast(tf.round(tf.constant((featureSize, featureSize), dtype=tf.float32) * randomScale), tf.int32)), featureSize, featureSize)[0]
        continue

    return pointcloud, newPointcloudIndices, newCorners, newHeatmaps, imageFeatures


def augmentFlipping(pointcloud, pointcloudIndices, corners, heatmaps, imageFeatures):
    width = WIDTH
    height = HEIGHT

    orientation = tf.cast(tf.random_uniform(shape=[1], maxval=4)[0], tf.int32)

    #if orientation == 0:
    #return pointcloud, pointcloudIndices, newCorners, heatmaps

    xsTarget = pointcloudIndices % width
    ysTarget = pointcloudIndices / width

    reverseChannelsY = tf.constant([-1, 2, 1, 0, 3, 7, 6, 5, 4, 10, 9, 8, 11, 12, 15, 14, 13, 16, 20, 19, 18, 17], dtype=tf.int32) + 1
    reverseChannelsX = tf.constant([-1, 0, 3, 2, 1, 5, 4, 7, 6, 8, 11, 10, 9, 12, 13, 16, 15, 14, 18, 17, 20, 19], dtype=tf.int32) + 1

    xsTarget = tf.cond(orientation >= 2, lambda: width - 1 - xsTarget, lambda: xsTarget)
    pointcloud = tf.cond(orientation >= 2, lambda: tf.concat([1 - pointcloud[:, 0:1], pointcloud[:, 1:]], axis=1), lambda: pointcloud)
    heatmaps = tf.cond(orientation >= 2, lambda: heatmaps[:, :, ::-1], lambda: heatmaps)
    corners = tf.cond(orientation >= 2, lambda: tf.stack([width - 1 - corners[:, 0], corners[:, 1], tf.gather_nd(reverseChannelsX, corners[:, 2:3])], axis=1), lambda: corners)

    ysTarget = tf.cond(tf.equal(orientation % 2, 1), lambda: height - 1 - ysTarget, lambda: ysTarget)
    pointcloud = tf.cond(tf.equal(orientation % 2, 1), lambda: tf.concat([pointcloud[:, :1], 1 - pointcloud[:, 1:2], pointcloud[:, 2:]], axis=1), lambda: pointcloud)
    heatmaps = tf.cond(tf.equal(orientation % 2, 1), lambda: heatmaps[:, ::-1], lambda: heatmaps)
    corners = tf.cond(tf.equal(orientation % 2, 1), lambda: tf.stack([corners[:, 0], height - 1 - corners[:, 1], tf.gather_nd(reverseChannelsY, corners[:, 2:3])], axis=1), lambda: corners)


    for index, (size, numChannels) in enumerate(zip(SIZES, NUM_CHANNELS)[1:]):
        if index in imageFeatures:
            imageFeatures[index] = tf.cond(orientation >= 2, lambda: imageFeatures[index][:, ::-1], lambda: imageFeatures[index])
            imageFeatures[index] = tf.cond(tf.equal(orientation % 2, 1), lambda: imageFeatures[index][::-1], lambda: imageFeatures[index])
        continue

    newPointcloudIndices = ysTarget * width + xsTarget

    return pointcloud, newPointcloudIndices, corners, heatmaps, imageFeatures


def augmentDropping(pointcloud, pointcloud_indices, changeIndices):
    p = tf.random.random() * 0.5 + 0.5
    indices = tf.range(pointcloud.shape[0], dtype='int32')
    out_shape = int(pointcloud.shape[0] * p)
    chosen_indices = tf.random.choice(indices, (out_shape, ), replace=True)
    rest_mask = tf.ones(indices.shape, dtype=tf.bool)
    rest_mask[chosen_indices] = 0
    rest_indices = indices[rest_mask]
    #rest_indices = tf.array(list(set(indices) - set(chosen_indices)))
    #rest = pointcloud[rest_indices]

    #rest_chosen_indices = tf.random.choice(rest_indices, (pointcloud.shape[0] - out_shape, ), replace=True)
    #rest_chosen = rest[rest_chosen_indices]
    #aug_pointcloud = pointcloud[chosen_indices] = rest_chosen

    rest_indices = tf.random.choice(rest_indices, chosen_indices.shape[0], replace=True)
    pointcloud[chosen_indices] = pointcloud[rest_indices]
    if changeIndices:
        pointcloud_indices[chosen_indices] = pointcloud_indices[rest_indices]
    return pointcloud, pointcloud_indices



def augment(pointcloud_inp, pointcloud_indices_0_inp, heatmapBatches, augmentation, numPoints=50000, numInputChannels=7):
    pointcloud_indices_inp = tf.zeros((FETCH_BATCH_SIZE, 6, NUM_POINTS),dtype='int32')
    newHeatmapBatches = [[] for heatmapIndex in xrange(len(heatmapBatches))]

    for imageIndex in xrange(pointcloud_inp.shape[0]):
        # pointcloud = pointcloud_inp[imageIndex]
        # pointcloud_indices_0 = pointcloud_indices_0_inp[imageIndex]
        # corner = corner_gt[imageIndex]
        # icon = icon_gt[imageIndex]
        # room = room_gt[imageIndex]
        # feature = feature_inp[imageIndex]
        # if 'w' in augmentation:
        #     pointcloud_indices_0, [corner, icon, room, feature] = augmentWarping(pointcloud_indices_0, [corner, icon, room, feature], gridStride=32., randomScale=4)
        # if 's' in augmentation:
        #     pointcloud_indices_0, [corner, icon, room, feature] = augmentScaling(pointcloud_indices_0, [corner, icon, room, feature], randomScale=0)
        # if 'f' in augmentation:
        #     pointcloud_indices_0, [corner, icon, room, feature] = augmentFlipping(pointcloud_indices_0, [corner, icon, room, feature])
        # if 'd' in augmentation:
        #     pointcloud, pointcloud_indices_0 = augmentDropping(pointcloud, pointcloud_indices_0, changeIndices=True)
        # if 'p' in augmentation:
        #     pointcloud, pointcloud_indices_0 = augmentDropping(pointcloud, pointcloud_indices_0, changeIndices=False)

        # pointcloud_inp[imageIndex] = pointcloud
        # pointcloud_indices_inp[imageIndex] = getCoarseIndicesMaps(pointcloud_indices_0, WIDTH, HEIGHT, 0)
        # corner_gt[imageIndex] = corner
        # icon_gt[imageIndex] = icon
        # room_gt[imageIndex] = room
        # feature_inp[imageIndex] = feature


        newHeatmaps = [heatmapBatch[imageIndex] for heatmapBatch in heatmapBatches]
        if 'w' in augmentation:
            pointcloud_indices_0_inp[imageIndex], newHeatmaps = augmentWarping(pointcloud_indices_0_inp[imageIndex], newHeatmaps, gridStride=32, randomScale=4)
        if 's' in augmentation:
            pointcloud_inp[imageIndex], pointcloud_indices_0_inp[imageIndex], newHeatmaps = augmentScaling(pointcloud_inp[imageIndex], pointcloud_indices_0_inp[imageIndex], newHeatmaps)
        if 'f' in augmentation:
            pointcloud_inp[imageIndex], pointcloud_indices_0_inp[imageIndex], newHeatmaps = augmentFlipping(pointcloud_inp[imageIndex], pointcloud_indices_0_inp[imageIndex], newHeatmaps)
        if 'd' in augmentation:
            pointcloud_inp[imageIndex], pointcloud_indices_0_inp[imageIndex] = augmentDropping(pointcloud_inp[imageIndex], pointcloud_indices_0_inp[imageIndex], changeIndices=True)
        if 'p' in augmentation:
            pointcloud_inp[imageIndex], pointcloud_indices_0_inp[imageIndex] = augmentDropping(pointcloud_inp[imageIndex], pointcloud_indices_0_inp[imageIndex], changeIndices=False)

        #print(pointcloud_indices_0_inp[imageIndex].shape, pointcloud_indices_inp[imageIndex].shape)
        pointcloud_indices_inp[imageIndex] = getCoarseIndicesMaps(pointcloud_indices_0_inp[imageIndex], WIDTH, HEIGHT, 0)
        for heatmapIndex, newHeatmap in enumerate(newHeatmaps):
            newHeatmapBatches[heatmapIndex].append(newHeatmap)
            continue
        continue
    newHeatmapBatches = [tf.array(newHeatmapBatch) for newHeatmapBatch in newHeatmapBatches]
    pointcloud_inp = tf.concatenate([pointcloud_inp, tf.ones((FETCH_BATCH_SIZE, NUM_POINTS, 1))], axis=2)
    #print(pointcloud_itf.shape)
    #writePointCloud('test/pointcloud.ply', pointcloud_inp[0, :, :6])
    #exit(1)

    if numPoints < pointcloud_itf.shape[1]:
        sampledInds = tf.range(pointcloud_itf.shape[1])
        tf.random.shuffle(sampledInds)
        sampledInds = sampledInds[:numPoints]
        pointcloud_inp = pointcloud_inp[:, sampledInds]
        pointcloud_indices_inp = pointcloud_indices_inp[:, :, sampledInds]

    if numInputChannels == 4:
        pointcloud_inp = tf.concatenate([pointcloud_inp[:, :, :3], pointcloud_inp[:, :, 6:]], axis=2)


    return pointcloud_inp, pointcloud_indices_inp, newHeatmapBatches
