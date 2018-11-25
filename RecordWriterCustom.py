import sys
import numpy as np
import cv2
import os
import tensorflow as tf

HEIGHT = 256
WIDTH = 256
NUM_POINTS = 50000
NUM_CHANNELS = [7, 64, 64, 64, 128, 256]
SIZES = [WIDTH, WIDTH // 2, WIDTH // 4, WIDTH // 8, WIDTH // 16, WIDTH // 32]

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]

class RecordWriterCustom():
    def __init__(self, filename_list, numPoints=50000, numInputChannels=6):
        """filename_list: a list of filename dictionaries Each element is a dict of filenames for data entries, which must contains 'point_cloud'
        numPoints: the number of sample points
        numInputChannels: [3: XYZ, 6:XYZ + RGB, 9: XYZ + RGB + Normals]
        """
        self.numPoints = numPoints
        self.numInputChannels = numInputChannels
        #super(RecordWriterTango, self).__init__()
        self.writer = tf.python_io.TFRecordWriter('data/' + filename + '_' + options.task + '.tfrecords')
        for filename_dict in filename_list:
            self.writeExample(filename_dict)
            continue
        self.writer.close()
        return
    

    def loadPoints(self, filename):
        """Function to load the point cloud"""
        assert(False, "Not implemented")
        return

    def rotatePoints(self, points, rotation_matrix=None):
        """Function to rotate the point cloud to be axis-aligned.
        Recommand annotate the rotation by adapting the annotator here: https://github.com/art-programmer/FloorplanAnnotator.
        If the annotation is not available we use heuristics to rotate the point cloud
        """
        #assert(False, "Not implemented")
        if transformation:
            rotated_points = points.copy()
            rotated_points[:, :3] = np.matmul(points[:, :3], rotation_matrix.transpose())
            if points.shape[-1] >= 9:
                rotated_points[:, 6:9] = np.matmul(points[:, 6:9], rotation_matrix.transpose())                
                pass
            return rotated_points
        
        if points.shape[-1] >= 9:
            ## Point normal exists
            normals = points[:, 6:9]
        else:
            ## Compute the point normal by ourselves. Consider using PCL if available. If not we sample some points and compute their normals
            sampled_points = points[np.random.choice(np.arange(len(points), dtype=np.int32), 100)]
            scale = (points[:, :3].max(0) - points[:, :3].min(0)).max()
            normals = []
            for sampled_point in sampled_points:
                neighbor_points = points[np.linalg.norm(points - sampled_point, axis=1) < scale * 0.02]
                if len(neighbor_points) >= 3:
                    ## Consider to use try catch for fitPlane as it might encounter singular cases
                    plane = fitPlane(neighbor_points)
                    normal = plane / max(np.linalg.norm(plane), 1e-4)
                    normals.append(normal)
                    pass
                continue
            normals = np.stack(normals, axis=0)
            pass
        

        polarAngles = np.arange(16) * np.pi / 2 / 16
        azimuthalAngles = np.arange(64) * np.pi * 2 / 64
        polarAngles = np.expand_dims(polarAngles, -1)
        azimuthalAngles = np.expand_dims(azimuthalAngles, 0)

        normalBins = np.stack([np.sin(polarAngles) * np.cos(azimuthalAngles), np.tile(np.cos(polarAngles), [1, azimuthalAngles.shape[1]]), -np.sin(polarAngles) * np.sin(azimuthalAngles)], axis=2)
        normalBins = np.reshape(normalBins, [-1, 3])
        numBins = normalBins.shape[0]
    
    
        normalDiff = np.tensordot(normals, normalBins, axes=([1], [1]))
        normalDiffSign = np.sign(normalDiff)
        normalDiff = np.maximum(normalDiff, -normalDiff)
        normalMask = one_hot(np.argmax(normalDiff, axis=-1), numBins)
        bins = normalMask.sum(0)
        maxNormals = np.expand_dims(normals, 1) * np.expand_dims(normalMask, -1)
        maxNormals *= np.expand_dims(normalDiffSign, -1)
        averageNormals = maxNormals.sum(0) / np.maximum(np.expand_dims(bins, -1), 1e-4)
        averageNormals /= np.maximum(np.linalg.norm(averageNormals, axis=-1, keepdims=True), 1e-4)
        dominantNormal_1 = averageNormals[np.argmax(bins)]

        dotThreshold_1 = np.cos(np.deg2rad(100))
        dotThreshold_2 = np.cos(np.deg2rad(80))
    
        dot_1 = np.tensordot(normalBins, dominantNormal_1, axes=([1], [0]))
        bins[np.logical_or(dot_1 < dotThreshold_1, dot_1 > dotThreshold_2)] = 0
        dominantNormal_2 = averageNormals[np.argmax(bins)]
        
        # dot_2 = np.tensordot(normalBins, dominantNormal_2, axes=([1], [0]))
        # bins[np.logical_or(dot_2 < dotThreshold_1, dot_2 > dotThreshold_2)] = 0    
        # dominantNormal_3 = averageNormals[np.argmax(bins)]
        dominantNormal_3 = np.cross(dominant_normal_1, dominant_normal_2)
        dominantNormal_2 = np.cross(dominant_normal_3, dominant_normal_1)        
        rotation_matrix = np.stack([dominantNormal_1, dominantNormal_2, dominantNormal_3], axis=0)
        rotated_points = points.copy()
        rotated_points[:, :3] = np.matmul(points[:, :3], rotation_matrix.transpose())
        if points.shape[-1] >= 9:
            rotated_points[:, 6:9] = np.matmul(points[:, 6:9], rotation_matrix.transpose())                
            pass        
        ## Rectify the rotation matrix
        return rotated_points

    def scalePoints(self, points, rotation_matrix=None):
        """Function to scale the point cloud to range [0, 1]."""    
        XYZ = points[:, :3]
        mins = XYZ.min(0, keepdims=True)
        maxs = XYZ.max(0, keepdims=True)
        maxRange = (maxs - mins)[:, :2].max()
        padding = maxRange * 0.05
        maxRange += padding * 2        
        mins = (maxs + mins) / 2 - maxRange / 2

        XYZ = (XYZ - mins) / maxRange        
        points[:, :3] = XYZ
        return points

    def computeCoordinates(self, points):
        """Compute the image coordinate for each point"""
        coordinates = np.minimum(np.maximum(np.round(points[:, :2] * imageSize).astype(np.int32), 0), imageSize - 1)
        coordinates = coordinates[:, 1] * WIDTH + coordinates[:, 0]
        return coordinates
    
    def load(self, name, filename):
        """Function to load info"""
        assert(False, "Not implemented")        
        return
    
    def writeExample(self, filename_dict):
        """Write one data entry"""
        assert('point_cloud' in filename_dict)

        ## Implement a function to load the point cloud as a numpy array of size [NxC] where N is the number of points and C is the number of channels
        points = loadPoints(filename_dict['point_cloud'])

        points = points[:, :self.numInputChannels]
        if len(points) < self.numPoints:
            indices = np.arange(len(points))
            points = np.concatenate([points, points[np.random.choice(indices, self.numPoints - len(points))]], axis=0)
        elif len(points) > self.numPoints:
            sampledInds = np.arange(len(points))
            points = points[np.random.choice(indices, self.numPoints)]
            pass

        points = self.rotatedPoints(points)
        points = self.scalePoints(points)
        
        imageSize = np.array([WIDTH, HEIGHT])
        indicesMap = self.computeCoordinates()

        info_dict = {}
        for info_name in ['corner_gt', 'icon_gt', 'room_gt', 'image_feature']:
            if info_name in filename_dict:
                info = self.load(info_name, filename_dict[info_name])
            else:
                if info_name == 'corner_gt':
                    info = np.zeros((1, 3))
                elif info_name in ['icon_gt', 'room_gt']:
                    info = np.zeros((HEIGHT, WIDTH), dtype=uint8)
                else:
                    info = np.zeros(sum([size * size * numChannels for size, numChannels in zip(SIZES, NUM_CHANNELS)[1:]]))                    
                    pass
                pass
            info_dict[info_name] = info
            continue
        
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_path': _bytes_feature(filename_dict['point_cloud']),
            'points': _float_feature(points.reshape(-1)),
            'point_indices': _int64_feature(indicesMap.reshape(-1)),
            'corner': _int64_feature(info['corner_gt'].reshape(-1)),
            'num_corners': _int64_feature([len(info['corner_gt'])]),
            'icon': _bytes_feature(info['icon_gt'].tostring()),
            'room': _bytes_feature(info['room_gt'].tostring()),
            'image': _float_feature(info['image_feature'].reshape(-1)),
            'flags': _int64_feature(np.zeros((2, dtype=np.int64))),
        }))
        self.writer.write(example.SerializeToString())
        return
    

if __name__ == "__main__":
    RecordWriterCustom()
    exit(1)
