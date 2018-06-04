import sys
import numpy as np
import cv2
import random
import math
import os
import time
import zlib
import socket
import threading
import Queue
import sys
import cPickle as pickle
import PIL
import json
import glob
import torchfile
import math
import os.path
import zipfile
import lupa
#from drnseg import DRNSeg
from utils import *
from floorplan_utils import *
import argparse
from skimage import measure
from augmentation import augment
import tensorflow as tf

#import show3d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


#BATCH_SIZE=16
FETCH_BATCH_SIZE=16
HEIGHT=256
WIDTH=256
NUM_POINTS = 50000
ROOT_FOLDER = '/home/chenliu/Projects/Data/Tango/'
#CORNER_POINTS_FOLDER = '../../../Data/Floorplan_syn/test_pntnetpp_new/data/oldway/real/combined_out/'

TEST_INDICES = [153, 102, 115, 57, 104, 156, 154, 310, 70, 134, 98, 8, 51, 76, 87, 36, 18, 106, 10, 73]


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def load_pnt(pnt_file):
    all_pnts = []
    texture_file = pnt_file.replace('.obj', '.mtl')
    texture_exists = os.path.exists(texture_file)
    print(pnt_file, texture_exists)
    if texture_exists:
        with open(texture_file) as f:
            for line in f:
                values = line.split(' ')
                if values[0] == 'map_Kd':
                    paths = texture_file.split('/')
                    paths[-1] = values[1].strip()
                    image_file = '/'.join(paths)
                    #print(image_file)
                    texture = cv2.imread(image_file)
                    texture_width = texture.shape[1]
                    texture_height = texture.shape[0]
                    pass
                continue
            pass
        pass

    with open(pnt_file) as f:
        vertex_index = 0
        for line in f:
            eles = line.split()
            if len(eles) == 0:
                continue
            if eles[0] == "v":
                if texture_exists:
                    pnt = [float(eles[1]), float(eles[2]), float(eles[3])]
                else:
                    assert len(eles) >= 7, str(vertex_index) + line
                    pnt = [float(eles[1]), float(eles[2]), float(eles[3]), float(eles[4]), float(eles[5]), float(eles[6])]
                    pass
                all_pnts.append(pnt)
            elif eles[0] == 'vt' and texture_exists:
                u = float(eles[1])
                v = float(eles[2])
                if u < 0 or u > 1 or v < 0 or v > 1:
                    color = [0, 0, 0]
                else:
                    u = int(min(np.round(u * texture_width), texture_width - 1))
                    v = int(min(np.round((1 - v) * texture_height), texture_height - 1))
                    color = texture[v][u]
                    color = color.astype(np.float32) / 255
                    pass
                all_pnts[vertex_index] += [color[2], color[1], color[0]]
                vertex_index += 1
                pass
            continue
        pass
    #if texture_exists:
    #writePointCloud('test/pointcloud.ply', all_pnts)
    #exit(1)
    # print(all_pnts)
    return all_pnts

class PntMapper:
    def __init__(self, start, size, newstart, newsize):
        self.start = start
        self.size = size
        self.newsize = newsize
        self.newstart=newstart
    def map_pnts(self, x, y):
#        print(self.size)
        return (x - self.start[0])/(self.size[0]-1)*(self.newsize[0]-1) + self.newstart[0], (y- self.start[1])/(self.size[1]-1)*(self.newsize[1]-1) + self.newstart[1]
def pnts_to_heatmap_transformation_matrix(pnts, is_binary, label, transform=None, img=None, bd_box=None):
    if not isinstance(pnts, np.ndarray):
        pnts = np.array(pnts)
    if pnts.shape[0] == 0:
        return None
    if transform is not None:
        if not isinstance(transform, np.ndarray):
            transform = np.array(transform)
        aug_pnts = np.ones((pnts.shape[0], 6))
        aug_pnts[:, :-1] = pnts
        prj_pnts = np.transpose(np.dot(transform, np.transpose(aug_pnts, (1, 0))), (1, 0))
        prj_pnts /= prj_pnts[:, -1][:,np.newaxis]
    else:
        prj_pnts = pnts
    if bd_box==None:
        x_max = prj_pnts[:, 0].max()
        x_min = prj_pnts[:, 0].min()
        y_max = prj_pnts[:, 1].max()
        y_min = prj_pnts[:, 1].min()
    else:
        x_min = bd_box[0]
        y_min = bd_box[1]
        x_max = bd_box[2]
        y_max = bd_box[3]
    #print(x_min, y_min, x_max, y_max)
    x_size = math.ceil(x_max - x_min + 1)
    y_size = math.ceil(y_max - y_min + 1)

    #print(x_min, y_min, x_size, y_size)
    mx_size = max(x_size, y_size)

    pnt_mapper = PntMapper((x_min, y_min), (x_size, y_size), (0,0), (HEIGHT, WIDTH))
    if img is None:
        img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for pnt in prj_pnts:
#        print(pnt)
        m_pnt = pnt_mapper.map_pnts(pnt[0], pnt[1])
        #if len(pnts) < 30:
        #    print(pnt, m_pnt, (x_min, y_min, x_max, y_max))
        try:
            if is_binary:
                img[int(round(m_pnt[1])), int(round(m_pnt[0]))] = label
            else:
                img[int(round(m_pnt[1])), int(round(m_pnt[0]))] += 1
        except Exception as e:
            print(pnt, m_pnt, e)
            raise e
    return img, (x_min, y_min, x_max, y_max)

def getZRotationMatrix(deg):
    return np.array([[math.cos(deg), -math.sin(deg), 0],
                        [-math.sin(deg), -math.cos(deg), 0],
                        [0, 0, 1]])

class RecordWriterTango():
    def __init__(self, options):
        #super(RecordWriterTango, self).__init__()
        self.dataPath = 'data/'
        self.stopped=False
        self.split = options.task
        self.imagePaths = self.loadImagePaths()

        self.numImages = len(self.imagePaths)
        self.numPoints = options.numPoints
        self.numInputChannels = options.numInputChannels

        self.labelMap = self.loadLabelMap()
        self.gap = 3

        self.roomLabelMap = getRoomLabelMap()

        # numTrainingImages = int(round(self.numImages * 0.9))
        # if options.task == 'train':
        #     self.imagePaths = self.imagePaths[:numTrainingImages]
        # else:
        #     self.imagePaths = self.imagePaths[numTrainingImages:]
        #     pass

        if options.task == 'train':
            self.imagePaths = [imagePath for imagePath in self.imagePaths if int(imagePath[5:]) not in TEST_INDICES]
        else:
            self.imagePaths = [imagePath for imagePath in self.imagePaths if int(imagePath[5:]) in TEST_INDICES]
            pass

        filename = os.path.basename(__file__).rstrip('.py')[12:]
        self.writer = tf.python_io.TFRecordWriter('data/' + filename + '_' + options.task + '.tfrecords')
        for imagePath in self.imagePaths:
            self.writeExample(imagePath)
            continue
        self.writer.close()
        return

    def getNumBatchesTesting(self):
        #print(self.numBatches)
        #print(self.numBatchesTraining)
        return self.numBatches - self.numBatchesTraining

    def loadImagePaths(self):
        scene_ids = []
        for scene_id in os.listdir(ROOT_FOLDER):
            annotation_filename = ROOT_FOLDER + scene_id + '/annotation/floorplan.txt'
            exists1 = os.path.exists(annotation_filename)
            filename_glob = ROOT_FOLDER + scene_id + '/' + 'dataset/mesh/*.obj'
            filename = list(glob.glob(filename_glob))
            exists2 = len(filename) > 0
            if exists1 and exists2:
                scene_ids.append(scene_id)
            else:
                print(scene_id)
                pass
            continue
        return scene_ids

    def loadLabelMap(self):
        roomMap = getRoomLabelMap()

        iconMap = getIconLabelMap()

        # iconMap['washing_basin'] = iconMap['washbasin']
        # iconMap['sofa'] = 6
        # iconMap['chair'] = 5
        # iconMap['TV'] = iconMap['tv']
        #for icon in ['cooking_counter', 'bathtub', 'toilet', 'washing_basin', 'sofa', 'cabinet', 'bed', 'table', 'desk', 'refrigerator', 'TV', 'entrance', 'chair']:

        labelMap = {}
        for icon in ['cooking_counter', 'bathtub', 'toilet', 'washing_basin', 'sofa', 'cabinet', 'bed', 'table', 'desk', 'refrigerator', 'TV', 'entrance', 'chair']:
            if icon not in iconMap:
                print(icon)
                exit(1)
            labelMap[icon] = ('icons', iconMap[icon])
            continue
        for room in ['living_room', 'kitchen', 'bedroom', 'bathroom', 'office', 'closet', 'balcony', 'corridor', 'dining_room', 'stairs']:
            if room not in roomMap:
                print(room)
                exit(1)
            labelMap[room] = ('rooms', roomMap[room])
            continue

        labelMap['door'] = 11
        labelMap['window'] = 12
        return labelMap

    def writeExample(self, scene_id):
        filename_glob = ROOT_FOLDER + scene_id + '/' + 'dataset/mesh/*.obj'
        filename = list(glob.glob(filename_glob))[0]
        points = load_pnt(filename)
        points = np.array(points)


        #segmentation = segmentation[sampledInds[:NUM_POINTS]]
        filename = ROOT_FOLDER + scene_id + '/annotation/metadata.t7'
        metadata = torchfile.load(filename)
        topDownViewTransformation = metadata["topDownTransformation"]

        #degree = metadata["topDownViewAngle"]
        rotMat = getZRotationMatrix(-metadata["topDownViewAngle"])
        #print(rotMat, topDownViewTransformation)
        #exit(1)

        #XYZ_rotated = np.transpose(np.dot(rotMat, np.transpose(points[:, :3])))
        #XYZ = np.tensordot(np.concatenate([points[:, :3], np.ones((points.shape[0], 1))], axis=1), topDownViewTransformation, axes=((1), (1)))
        #XYZ[:, 2] = points[:, 2]

        #ratio_1 = (XYZ[:, 0].max() - XYZ[:, 0].min()) / (XYZ_rotated[:, 0].max() - XYZ_rotated[:, 0].min())
        #ratio_2 = (XYZ[:, 1].max() - XYZ[:, 1].min()) / (XYZ_rotated[:, 1].max() - XYZ_rotated[:, 1].min())

        #XYZ[2, 2] *= np.sqrt(ratio_1 * ratio_2)

        #ratio = pow(np.abs(rotMat[0][0] / topDownViewTransformation[0][0] * rotMat[1][0] / topDownViewTransformation[1][0] * rotMat[0][1] / topDownViewTransformation[0][1] * rotMat[1][1] / topDownViewTransformation[1][1]), 0.25)
        ratio = 0
        for i in xrange(2):
            for j in xrange(2):
                if rotMat[i][j] != 0:
                    ratio = max(ratio, np.abs(topDownViewTransformation[i][j] / rotMat[i][j]))
                    pass
                continue
            continue

        globalTransformation = topDownViewTransformation
        globalTransformation[2, 2] = ratio
        globalTransformation[2, 3] = 0
        globalTransformation = np.concatenate([globalTransformation, np.zeros((1, 4))], axis=0)
        globalTransformation[3, 3] = 1


        XYZ = np.tensordot(np.concatenate([points[:, :3], np.ones((points.shape[0], 1))], axis=1), globalTransformation, axes=((1), (1)))
        XYZ = XYZ[:, :3] / XYZ[:, 3:]


        mins = XYZ.min(0, keepdims=True)
        maxs = XYZ.max(0, keepdims=True)
        maxRange = (maxs - mins)[:, :2].max()
        padding = maxRange * 0.05
        mins = (maxs + mins) / 2 - maxRange / 2
        mins -= padding
        maxRange += padding * 2
        minXY = mins[:, :2]

        #XYZ[:, :2] = (XYZ[:, :2] - minXY) / maxRange
        XYZ = (XYZ - mins) / maxRange
        points[:, :3] = XYZ

        originalWidth = 700.

        if points.shape[0] < NUM_POINTS:
            indices = np.arange(points.shape[0])
            points = np.concatenate([points, points[np.random.choice(indices, NUM_POINTS - points.shape[0])]], axis=0)
        elif points.shape[0] > NUM_POINTS:
            sampledInds = np.arange(points.shape[0])
            np.random.shuffle(sampledInds)
            points = points[sampledInds[:NUM_POINTS]]
            pass

        points[:, 3:] = points[:, 3:] / 255 - 0.5

        coordinates = np.clip(np.round(points[:, :2] * HEIGHT).astype(np.int32), 0, HEIGHT - 1)

        self.indicesMaps = np.zeros((NUM_POINTS), dtype=np.int64)
        self.projectIndices(np.concatenate([coordinates, np.arange(NUM_POINTS).reshape(-1, 1)], axis=1), 0, WIDTH, 0, HEIGHT)


        filename = ROOT_FOLDER + scene_id + '/annotation/floorplan.txt'
        walls = []
        doors = []
        windows = []
        semantics = {}
        def transformPoint(v, c):
            return max(min(round((float(v) - minXY[0, c]) / maxRange * WIDTH), WIDTH - 1), 0)

        with open(filename) as info_file:
            line_index = 0
            for line in info_file:
                line = line.split('\t')
                if line[4] == 'wall':
                    walls.append(((transformPoint(line[0], 0), transformPoint(line[1], 1)), (transformPoint(line[2], 0), transformPoint(line[3], 1))))
                elif line[4] == 'door':
                    doors.append(((transformPoint(line[0], 0), transformPoint(line[1], 1)), (transformPoint(line[2], 0), transformPoint(line[3], 1))))
                elif line[4] == 'window':
                    windows.append(((transformPoint(line[0], 0), transformPoint(line[1], 1)), (transformPoint(line[2], 0), transformPoint(line[3], 1))))
                else:
                    if line[4] not in semantics:
                        semantics[line[4]] = []
                        pass
                    semantics[line[4]].append(((transformPoint(line[0], 0), transformPoint(line[1], 1)), (transformPoint(line[2], 0), transformPoint(line[3], 1))))
                    pass
                continue
            pass


        roomSegmentation = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for line in walls:
            cv2.line(roomSegmentation, (int(round(line[0][0])), int(round(line[0][1]))), (int(round(line[1][0])), int(round(line[1][1]))), color = 15 + calcLineDirection(line), thickness=self.gap)
            #cv2.line(roomSegmentation, (int(round(line[0][0])), int(round(line[0][1]))), (int(round(line[1][0])), int(round(line[1][1]))), color = 15, thickness=self.gap)
            continue

        rooms = measure.label(roomSegmentation == 0, background=0)


        corners = lines2Corners(walls, gap=self.gap)
        #corner_gt = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        corner_gt = []
        for corner in corners:
            #corner_gt[int(round(corner[0][1])), int(round(corner[0][0]))] = corner[1] + 1
            corner_gt.append((int(round(corner[0][0])), int(round(corner[0][1])), corner[1] + 1))
            continue

        openingCornerMap = [[3, 1], [0, 2]]
        for openingType, openings in enumerate([doors, windows]):
            for opening in openings:
                direction = calcLineDirection(opening)
                for cornerIndex, corner in enumerate(opening):
                    #corner_gt[int(round(corner[1])), int(round(corner[0]))] = 14 + openingCornerMap[direction][cornerIndex]
                    corner_gt.append((int(round(corner[0])), int(round(corner[1])), 14 + openingCornerMap[direction][cornerIndex]))
                    continue
                continue
            continue


        wallIndex = rooms.min()
        for pixel in [(0, 0), (0, HEIGHT - 1), (WIDTH - 1, 0), (WIDTH - 1, HEIGHT - 1)]:
            backgroundIndex = rooms[pixel[1]][pixel[0]]
            if backgroundIndex != wallIndex:
                break
            continue

        iconSegmentation = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        for line in doors:
            cv2.line(iconSegmentation, (int(round(line[0][0])), int(round(line[0][1]))), (int(round(line[1][0])), int(round(line[1][1]))), color = self.labelMap['door'], thickness=self.gap - 1)
            continue
        for line in windows:
            cv2.line(iconSegmentation, (int(round(line[0][0])), int(round(line[0][1]))), (int(round(line[1][0])), int(round(line[1][1]))), color = self.labelMap['window'], thickness=self.gap - 1)
            continue

        roomLabelMap = {}
        for semantic, items in semantics.iteritems():
            group, label = self.labelMap[semantic]
            for corners in items:
                if group == 'icons':
                    if label == 0:
                        continue
                    cv2.rectangle(iconSegmentation, (int(round(corners[0][0])), int(round(corners[0][1]))), (int(round(corners[1][0])), int(round(corners[1][1]))), color=label, thickness=-1)
                    # corner_gt[int(round(corners[0][1])), int(round(corners[0][0]))] = 18 + 2
                    # corner_gt[int(round(corners[1][1])), int(round(corners[0][0]))] = 18 + 1
                    # corner_gt[int(round(corners[0][1])), int(round(corners[1][0]))] = 18 + 3
                    # corner_gt[int(round(corners[1][1])), int(round(corners[1][0]))] = 18 + 0
                    corner_gt.append((int(round(corners[0][0])), int(round(corners[0][1])), 18 + 2))
                    corner_gt.append((int(round(corners[0][0])), int(round(corners[1][1])), 18 + 1))
                    corner_gt.append((int(round(corners[1][0])), int(round(corners[0][1])), 18 + 3))
                    corner_gt.append((int(round(corners[1][0])), int(round(corners[1][1])), 18 + 0))
                else:
                    roomIndex = rooms[int(round((corners[0][1] + corners[1][1]) / 2))][int(round((corners[0][0] + corners[1][0]) / 2))]
                    if roomIndex == wallIndex or roomIndex == backgroundIndex:
                        print('label on background')
                        exit(1)
                        pass
                    if roomIndex in roomLabelMap:
                        print('room has more than one labels', label)
                        exit(1)
                        pass
                    roomLabelMap[roomIndex] = label
                    roomSegmentation[rooms == roomIndex] = label
                    pass
                continue
            continue
        for roomIndex in xrange(rooms.min(), rooms.max() + 1):
            if roomIndex == wallIndex or roomIndex == backgroundIndex:
                continue
            if roomIndex not in roomLabelMap:
                print('room has no label')
                print(roomIndex, rooms.max())
                pass
            continue
        flags = np.zeros(2, np.int64)
        flags[0] = 1

        corner_feature_file =  ROOT_FOLDER + scene_id + '/corner_acc.npy'
        icon_feature_file =  ROOT_FOLDER + scene_id + '/topdown_acc.npy'
        image_features = [[], []]
        if os.path.exists(corner_feature_file) and os.path.exists(icon_feature_file):
            flags[1] = 1
            corner_feature = np.load(corner_feature_file).reshape((HEIGHT, WIDTH, -1))
            icon_feature = np.load(icon_feature_file).reshape((HEIGHT, WIDTH, -1))
            for featureIndex, feature in enumerate([corner_feature, icon_feature]):
                for size, numChannels in zip(SIZES, NUM_CHANNELS)[1:]:
                    feature = cv2.resize(feature, (size, size))
                    image_features[featureIndex].append(feature.reshape((size, size, numChannels, -1)).mean(-1).reshape(-1))
                    continue
                image_features[featureIndex] = np.concatenate(image_features[featureIndex], axis=0)
                continue
            image_features = image_features[0] + image_features[1]
        else:
            image_features = np.zeros(sum([size * size * numChannels for size, numChannels in zip(SIZES, NUM_CHANNELS)[1:]]))
            pass

        if False:
            cv2.imwrite('test/density.png', drawDensityImage(getDensity(points, HEIGHT, WIDTH)))
            cv2.imwrite('test/density_indices.png', drawDensityImage(getDensityFromIndices(self.indicesMaps, HEIGHT, WIDTH)))
            cv2.imwrite('test/icon_segmentation.png', drawSegmentationImage(iconSegmentation))
            cv2.imwrite('test/room_segmentation.png', drawSegmentationImage(roomSegmentation))
            cv2.imwrite('test/corner_segmentation.png', drawSegmentationImage(corner_gt, blackIndex=0))
            if flags[1]:
                cv2.imwrite('test/topdown_corner.png', cv2.imread(ROOT_FOLDER + scene_id + '/corner_pred.png'))
                cv2.imwrite('test/topdown_icon.png', cv2.imread(ROOT_FOLDER + scene_id + '/topdown_pred_nonzero.png'))
                exit(1)
                pass
            pass

        corner_gt = np.array(corner_gt, dtype=np.int64)
        numCorners = len(corner_gt)
        print('num corners', numCorners)
        if numCorners > MAX_NUM_CORNERS:
            exit(1)
        elif numCorners < MAX_NUM_CORNERS:
            corner_gt = np.concatenate([corner_gt, np.zeros((MAX_NUM_CORNERS - numCorners, 3), dtype=np.int64)], axis=0)
            pass

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_path': _bytes_feature(scene_id),
            'points': _float_feature(points.reshape(-1)),
            'point_indices': _int64_feature(self.indicesMaps.reshape(-1)),
            'corner': _int64_feature(corner_gt.reshape(-1)),
            'num_corners': _int64_feature([numCorners]),
            'icon': _bytes_feature(iconSegmentation.tostring()),
            'room': _bytes_feature(roomSegmentation.tostring()),
            'image': _float_feature(image_features.reshape(-1)),
            'flags': _int64_feature(flags),
        }))
        self.writer.write(example.SerializeToString())
        return


    def projectSegmentation(self, pointSegmentation, min_x, max_x, min_y, max_y):
        if max_x - min_x == 1 and max_y - min_y == 1:
            segments, counts = np.unique(pointSegmentation[:, 2], return_counts=True)
            segmentList = zip(segments.tolist(), counts.tolist())
            segmentList = [segment for segment in segmentList if segment[0] not in [0, 2]]
            label = 0
            if 2 in segments:
                label = 2
                pass
            if len(segmentList) > 0:
                segment = max(segmentList, key=lambda x: x[1])
                if segment[1] > 0:
                    label = segment[0]
                    pass
                pass
            self.segmentation[min_y][min_x] = label
        elif max_x - min_x >= max_y - min_y:
            middle_x = int((max_x + min_x + 1) / 2)
            mask_1 = pointSegmentation[:, 1] < middle_x
            self.projectSegmentation(pointSegmentation[mask_1], min_x, middle_x, min_y, max_y)
            mask_2 = pointSegmentation[:, 1] >= middle_x
            self.projectSegmentation(pointSegmentation[mask_2], middle_x, max_x, min_y, max_y)
        else:
            middle_y = int((max_y + min_y + 1) / 2)
            mask_1 = pointSegmentation[:, 0] < middle_y
            self.projectSegmentation(pointSegmentation[mask_1], min_x, max_x, min_y, middle_y)
            mask_2 = pointSegmentation[:, 0] >= middle_y
            self.projectSegmentation(pointSegmentation[mask_2], min_x, max_x, middle_y, max_y)
            pass
        return


    def projectIndices(self, pointSegmentation, min_x, max_x, min_y, max_y):
        # for strideIndex in xrange(6):
        #     stride = pow(2, strideIndex)
        #     if max_x - min_x == stride and max_y - min_y == stride:
        #         self.indicesMaps[strideIndex][pointSegmentation[:, 2]] = min_y / stride * WIDTH / stride + min_x / stride + self.batchIndex * (HEIGHT / stride * WIDTH / stride)
        #         pass
        #     continue

        if max_x - min_x == 1 and max_y - min_y == 1:
            self.indicesMaps[pointSegmentation[:, 2]] = min_y * WIDTH + min_x
            return
        elif max_x - min_x >= max_y - min_y:
            middle_x = int((max_x + min_x + 1) / 2)
            mask_1 = pointSegmentation[:, 0] < middle_x
            self.projectIndices(pointSegmentation[mask_1], min_x, middle_x, min_y, max_y)
            mask_2 = pointSegmentation[:, 0] >= middle_x
            self.projectIndices(pointSegmentation[mask_2], middle_x, max_x, min_y, max_y)
        else:
            middle_y = int((max_y + min_y + 1) / 2)
            mask_1 = pointSegmentation[:, 1] < middle_y
            self.projectIndices(pointSegmentation[mask_1], min_x, max_x, min_y, middle_y)
            mask_2 = pointSegmentation[:, 1] >= middle_y
            self.projectIndices(pointSegmentation[mask_2], min_x, max_x, middle_y, max_y)
            pass
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Planenet')
    args = parser.parse_args()
    args.numPoints = 50000
    args.numInputChannels = 7
    args.augmentation = ''
    for split in ['val']:
        print(split)
        args.task = split
        RecordWriterTango(args)
        continue
    exit(1)

# reader = BatchFetcher('train')
# for i in xrange(12):
#     print(i)
#     reader.work(i)
#     continue
# exit(1)
