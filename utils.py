import numpy as np
import glob
import cv2
if 'xrange' not in globals():
    xrange = range

#consistent color mapping
class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(1)
        #self.colorMap = np.random.randint(255, size = (numColors, 3), dtype=np.uint8)
        #self.colorMap[0] = 0
        #self.colorMap[0] = np.maximum(self.colorMap[0], 1)

        # [128, 128, 128],
        # [0, 0, 255],
        # [64, 128, 192],
        # [0, 128, 0],
        # [192, 0, 0],
        # [128, 0, 128],
        # [128, 128, 192],
        # [128, 192, 192],
        # [0, 128, 0],
        # [0, 0, 128],
        # [128, 128, 0],
        # [0, 128, 128]

        self.colorMap = np.array([[255, 0, 0],
                                  [50, 150, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [255, 255, 0],
                                  [0, 255, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [128, 0, 255],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ], dtype=np.uint8)

        self.colorMap = np.maximum(self.colorMap, 1)

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors, 3), dtype=np.uint8)], axis=0)
            pass

        return

    def getColorMap(self):
        return self.colorMap

    def getColor(self, index):
        if index >= colorMap.shape[0]:
            return np.random.randint(255, size = (3), dtype=np.uint8)
        else:
            return self.colorMap[index]
            pass

def sigmoid(values):
    return 1 / (1 + np.exp(-values))

def softmax(values):
    values_exp = np.exp(values)
    return values_exp / np.sum(values_exp, axis=-1, keepdims=True)


#Draw density image
def drawDensityImage(density, maxDensity=-1, nChannels=1):
    if maxDensity < 0:
        maxDensity = density.max() / 2
        pass
    densityImage = np.minimum(np.round(density / maxDensity * 255).astype(np.uint8), 255)
    if nChannels == 3:
        densityImage = np.stack([densityImage, densityImage, densityImage], axis=2)
        pass
    return densityImage

#Draw segmentation image. The input could be either HxW or HxWxC
def drawSegmentationImage(segmentations, numColors=42, blackIndex=-1):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        #segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass
    segmentation = segmentation.astype(np.int)
    return randomColor[segmentation.reshape(-1)].reshape((height, width, 3))

def drawMaskImage(mask):
    return (np.clip(mask * 255, 0, 255)).astype(np.uint8)



def projectIndices(indicesMap, pointSegmentation, min_x, max_x, min_y, max_y, width):
    if max_x - min_x == 1 and max_y - min_y == 1:
        indicesMap[pointSegmentation[:, 2]] = min_y * width + min_x
        return
    elif max_x - min_x >= max_y - min_y:
        middle_x = int((max_x + min_x + 1) / 2)
        mask_1 = pointSegmentation[:, 0] < middle_x
        projectIndices(indicesMap, pointSegmentation[mask_1], min_x, middle_x, min_y, max_y, width)
        mask_2 = pointSegmentation[:, 0] >= middle_x
        projectIndices(indicesMap, pointSegmentation[mask_2], middle_x, max_x, min_y, max_y, width)
    else:
        middle_y = int((max_y + min_y + 1) / 2)
        mask_1 = pointSegmentation[:, 1] < middle_y
        projectIndices(indicesMap, pointSegmentation[mask_1], min_x, max_x, min_y, middle_y, width)
        mask_2 = pointSegmentation[:, 1] >= middle_y
        projectIndices(indicesMap, pointSegmentation[mask_2], min_x, max_x, middle_y, max_y, width)
        pass
    return


#Extract corners from heatmaps
def extractCornersFromHeatmaps(heatmaps, heatmapThreshold=0.5, numPixelsThreshold=5, returnRanges=True):
    from skimage import measure
    heatmaps = (heatmaps > heatmapThreshold).astype(np.float32)
    orientationPoints = []
    #kernel = np.ones((3, 3), np.float32)
    for heatmapIndex in xrange(0, heatmaps.shape[-1]):
        heatmap = heatmaps[:, :, heatmapIndex]
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in xrange(components.min() + 1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            if ys.shape[0] <= numPixelsThreshold:
                continue
            #print(heatmapIndex, xs.shape, ys.shape, componentIndex)
            if returnRanges:
                points.append(((xs.mean(), ys.mean()), (xs.min(), ys.min()), (xs.max(), ys.max())))
            else:
                points.append((xs.mean(), ys.mean()))
                pass
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

#Extract corners from heatmaps
def extractCornersFromSegmentation(segmentation, cornerTypeRange=[0, 13]):
    from skimage import measure
    orientationPoints = []
    for heatmapIndex in xrange(cornerTypeRange[0], cornerTypeRange[1]):
        heatmap = segmentation == heatmapIndex
        #heatmap = cv2.dilate(cv2.erode(heatmap, kernel), kernel)
        components = measure.label(heatmap, background=0)
        points = []
        for componentIndex in xrange(components.min()+1, components.max() + 1):
            ys, xs = (components == componentIndex).nonzero()
            points.append((xs.mean(), ys.mean()))
            continue
        orientationPoints.append(points)
        continue
    return orientationPoints

#Extract corners from heatmaps
def getSegmentationFromCorners(width, height, orientationCorners):
    segmentation = np.zeros((height, width))
    for orientation, corners in enumerate(orientationCorners):
        for corner in corners:
            segmentation[int(round(corner[1]))][int(round(corner[0]))] = orientation + 1
            continue
        continue
    return segmentation

#Evaluate corner predictions
def evaluateCorners(cornersPred, cornersGT, distanceThreshold = 15):
    numGT = 0
    numPred = 0
    numMatches = 0
    for cornerType, gt_c in enumerate(cornersGT):
        pred_c = cornersPred[cornerType]
        gt_c = np.array(gt_c)
        pred_c = np.array(pred_c)
        numGT += gt_c.shape[0]
        numPred += pred_c.shape[0]
        if gt_c.shape[0] == 0 or pred_c.shape[0] == 0:
            continue
        diff = np.linalg.norm(np.expand_dims(gt_c, 1) - np.expand_dims(pred_c, 0), axis=2)
        numMatches += (diff.min(axis=1) < distanceThreshold).sum()
        continue
    return np.array([numMatches, numGT, numPred])


#Evaluate segmentation predictions
def evaluateSegmentation(segmentationPred, segmentationGT, numSegments = 12):
    #print("hack in evaluate! remove this!")
    #print(segmentationGT.shape)
    #segmentationGT = segmentationGT.transpose(1, 0)
    height = segmentationPred.shape[0]
    width = segmentationPred.shape[1]
    nonemptyMask = segmentationGT > 0
    correctMask = segmentationPred == segmentationGT
    #accuracy = float(correctMask.sum()) / (width * height)
    accuracy = float(correctMask[nonemptyMask].sum()) / max(nonemptyMask.sum(), 1)
    #(width * height)

    sumIOU = 0.
    numIOU = 0
    for segmentIndex in xrange(numSegments):
        gt_s = segmentationGT == segmentIndex
        pred_s = segmentationPred == segmentIndex
        union = np.logical_or(pred_s, gt_s)
        unionSum = union.sum()
        if unionSum == 0:
            continue
        intersection = np.logical_and(pred_s, gt_s)
        IOU = float(intersection.sum()) / unionSum
        sumIOU += IOU
        numIOU += 1
        continue
    meanIOU = sumIOU / numIOU
    return np.array([accuracy, meanIOU])

def evaluateDetection(segmentationPred, segmentationGT, numSegments = 12, IOUThreshold = 0.5):
    from skimage import measure
    numGT = 0
    numPred = 0
    numMatches = 0

    for segmentIndex in xrange(numSegments):
        gt_s = segmentationGT == segmentIndex
        if gt_s.sum() > 0:
           numGT += 1
           pass
        pred_s = segmentationPred == segmentIndex
        if pred_s.sum() > 0:
           numPred += 1
           pass
        IOU = float(np.logical_and(pred_s, gt_s).sum()) / np.logical_or(pred_s, gt_s).sum()
        if IOU > IOUThreshold:
            numMatches += 1
            pass
        continue
    return (numMatches, numGT, numPred)

def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]

def rotatePoints(points, segmentation, numSampledPoints = 10000):
    sampledInds = np.arange(points.shape[0])
    np.random.shuffle(sampledInds)
    sampledPoints = points[sampledInds[:numSampledPoints]]
    sampledSegmentation = segmentation[sampledInds[:numSampledPoints]]
    segments = np.unique(sampledSegmentation).tolist()
    binSize = 3
    numAngleBins = 90 // 3 + 1
    angleSums = np.zeros(numAngleBins)
    angleCounts = np.zeros(numAngleBins)
    for segmentIndex in segments:
        segmentPoints = sampledPoints[sampledSegmentation == segmentIndex]
        if segmentPoints.shape[0] < 3:
            continue
        try:
            plane = fitPlane(segmentPoints)
        except:
            continue
        if np.argmax(np.abs(plane)) == 2:
            continue
        angle = np.arctan2(plane[0], plane[1])
        angle = np.rad2deg(angle) % 90
        numPoints = segmentPoints.shape[0]
        angleSums[int(np.round(angle / 3))] += angle * numPoints
        angleCounts[int(np.round(angle / 3))] += numPoints
        continue
    angles = angleSums / np.maximum(angleCounts, 1)
    angle = angles[np.argmax(angleCounts)]
    angle = np.deg2rad(angle)
    rotationMatrix = np.zeros((2, 2))
    rotationMatrix[0][0] = np.cos(angle)
    rotationMatrix[0][1] = np.sin(angle)
    rotationMatrix[1][0] = -np.sin(angle)
    rotationMatrix[1][1] = np.cos(angle)

    points[:, :2] = np.matmul(points[:, :2], rotationMatrix)
    return points

def rotatePointsWithMatrix(points, segmentation, numSampledPoints = 10000):
    sampledInds = np.arange(points.shape[0])
    np.random.shuffle(sampledInds)
    sampledPoints = points[sampledInds[:numSampledPoints]]
    sampledSegmentation = segmentation[sampledInds[:numSampledPoints]]
    segments = np.unique(sampledSegmentation).tolist()
    binSize = 3
    numAngleBins = 90 // 3 + 1
    #print("rotate!", numAngleBins, flush=True)
    assert isinstance(numAngleBins, int), numAngleBins

    angleSums = np.zeros(numAngleBins)
    angleCounts = np.zeros(numAngleBins)
    for segmentIndex in segments:
        segmentPoints = sampledPoints[sampledSegmentation == segmentIndex]
        if segmentPoints.shape[0] < 3:
            continue
        try:
            plane = fitPlane(segmentPoints)
        except:
            continue
        if np.argmax(np.abs(plane)) == 2:
            continue
        angle = np.arctan2(plane[0], plane[1])
        angle = np.rad2deg(angle) % 90
        numPoints = segmentPoints.shape[0]
        angleSums[int(np.round(angle / 3))] += angle * numPoints
        angleCounts[int(np.round(angle / 3))] += numPoints
        continue
    angles = angleSums / np.maximum(angleCounts, 1)
    angle = angles[np.argmax(angleCounts)]
    angle = np.deg2rad(angle)
    rotationMatrix = np.zeros((2, 2))
    rotationMatrix[0][0] = np.cos(angle)
    rotationMatrix[0][1] = np.sin(angle)
    rotationMatrix[1][0] = -np.sin(angle)
    rotationMatrix[1][1] = np.cos(angle)

    points[:, :2] = np.matmul(points[:, :2], rotationMatrix)
    return points, rotationMatrix


def drawTopDownView(points, width, height):
    coordinates = points[:, :2]
    mins = coordinates.min(0, keepdims=True)
    maxs = coordinates.max(0, keepdims=True)
    ranges = maxs - mins
    padding = ranges * 0.05
    mins -= padding
    ranges += padding * 2

    maxRange = ranges.max()
    mins = (maxs + mins) / 2 - maxRange / 2

    coordinates = ((coordinates - mins) / ranges * height).astype(np.int32)
    coordinates = np.minimum(coordinates, height - 1)
    image = np.zeros((height, width))
    for coordinate in coordinates:
        image[coordinate[1]][coordinate[0]] += 1
        continue
    print(image.max())
    image /= min(image.max(), 300)
    image = (np.minimum(image * 255, 255)).astype(np.uint8)
    return image

def writePointCloud(filename, pointCloud):
    with open(filename, 'w') as f:
        header = """ply
format ascii 1.0
element vertex """
        header += str(len(pointCloud))
        header += """
property float x
property float y
property float z
property uchar red                                     { start of vertex color }
property uchar green
property uchar blue
end_header
"""
        f.write(header)
        for point in pointCloud:
            for valueIndex, value in enumerate(point):
                if valueIndex < 3:
                    f.write(str(value) + ' ')
                else:
                    f.write(str(int(value * 255)) + ' ')
                    pass
                continue
            f.write('\n')
            continue
        f.close()
        pass
    return


# def convertPointCloudIndices(folder, width=256, height=256):
#     filenames = glob.glob(folder + '/pointcloud_indices_*.npy')
#     for filename in filenames:
#         pointcloud_indices = np.load(filename)
#         new_pointcloud_indices = []
#         for imageIndex in xrange(pointcloud_indices.shape[0]):
#             pointcloud_indices_0 = pointcloud_indices[imageIndex][0] - imageIndex * width * height
#             new_pointcloud_indices.append(pointcloud_indices_0)
#             continue
#         np.save(filename, np.stack(new_pointcloud_indices, 0))
#         continue
#     return


def getDensity(points, width=256, height=256):
    imageSizes = np.array([width, height]).reshape((-1, 2))
    # mins = points.min(0, keepdims=True)
    # maxs = points.max(0, keepdims=True)
    # maxRange = (maxs - mins)[:, :2].max()
    # padding = maxRange * 0.05
    # mins = (maxs + mins) / 2 - maxRange / 2
    # mins -= padding
    # maxRange += padding * 2
    # coordinates = np.round((points - mins) / maxRange * imageSizes).astype(np.int32)
    coordinates = np.round(points[:, :2] * imageSizes).astype(np.int32)
    coordinates = np.minimum(np.maximum(coordinates, 0), imageSizes - 1)
    density = np.zeros((height, width))
    for uv in coordinates:
        density[uv[1], uv[0]] += 1
        continue
    return density

def getDensityFromIndices(indices, width=256, height=256):
    density = np.zeros((height, width))
    for index in indices:
        #print(index, index / width, index % width)
        density[index / width, index % width] += 1
        continue
    return density

def drawCornerImages(segmentations, numColors=42, blackIndex=0):
    if segmentations.ndim == 2:
        numColors = max(numColors, segmentations.max() + 2)
    else:
        numColors = max(numColors, segmentations.shape[2] + 2)
        pass
    randomColor = ColorPalette(numColors).getColorMap()
    if blackIndex >= 0:
        randomColor[blackIndex] = 0
        pass
    width = segmentations.shape[1]
    height = segmentations.shape[0]
    if segmentations.ndim == 3:
        #segmentation = (np.argmax(segmentations, 2) + 1) * (np.max(segmentations, 2) > 0.5)
        segmentation = np.argmax(segmentations, 2)
    else:
        segmentation = segmentations
        pass
    segmentation = segmentation.astype(np.int32).reshape(-1)
    images = []
    print(np.unique(segmentation))
    for segment in [(1, 14), (14, 18), (18, 22)]:
        colorMap = randomColor.copy()
        colorMap[:segment[0]] = 0
        colorMap[segment[1]:] = 0
        image = colorMap[segmentation].reshape((height, width, 3))
        image = cv2.dilate(image, np.ones((3, 3), dtype=np.uint8), iterations=3)
        images.append(image)
        continue
    return images

def segmentation2Heatmaps(segmentation, numLabels):
    width = segmentation.shape[1]
    height = segmentation.shape[0]
    labels = np.arange(numLabels, dtype=np.int32).reshape((1, 1, -1))
    heatmaps = (np.expand_dims(segmentation, -1) == labels).astype(np.float32)
    return heatmaps

def heatmaps2Segmentation(heatmaps):
    return np.argmax(heatmaps, axis=2)

def calcIOU(rectangle_1, rectangle_2):
    # mins_1 = rectangle_1.min(0)
    # maxs_1 = rectangle_1.max(0)
    # area_1 = (maxs_1[0] - mins_1[0] + 1) * (maxs_1[1] - mins_1[1] + 1)
    # mins_2 = rectangle_2.min(0)
    # maxs_2 = rectangle_2.max(0)
    # area_2 = (maxs_2[0] - mins_2[0] + 1) * (maxs_2[1] - mins_2[1] + 1)
    # intersection = (min(maxs_1[0], maxs_2[0]) - max(mins_1[0], mins_2[0]) + 1) * (min(maxs_1[1], maxs_2[1]) - max(mins_1[1], mins_2[1]) + 1)

    rectangles = [rectangle_1, rectangle_2]

    x_1 = max([int(round((rectangle[0][0] + rectangle[2][0]) / 2)) for rectangle in rectangles])
    x_2 = min([int(round((rectangle[1][0] + rectangle[3][0]) / 2)) for rectangle in rectangles])
    y_1 = max([int(round((rectangle[0][1] + rectangle[1][1]) / 2)) for rectangle in rectangles])
    y_2 = min([int(round((rectangle[2][1] + rectangle[3][1]) / 2)) for rectangle in rectangles])
    if x_1 >= x_2 or y_1 >= y_2:
        return 0
    intersection = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)

    area_1, area_2 = ((int(round((rectangle[1][0] + rectangle[3][0]) / 2)) - int(round((rectangle[0][0] + rectangle[2][0]) / 2)) + 1) * (int(round((rectangle[2][1] + rectangle[3][1]) / 2)) - int(round((rectangle[0][1] + rectangle[1][1]) / 2)) + 1) for rectangle in rectangles)

    union = area_1 + area_2 - intersection
    return float(intersection) / union

def calcIOUMask(mask_1, mask_2):
    intersection = (mask_1 * mask_2).sum()
    union = mask_1.sum() + mask_2.sum() - intersection
    return float(intersection) / max(union, 1)


def gaussian(k=5, sig=0):
    """
    creates gaussian kernel with side length l and a sigma of sig
v    """
    if sig == 0:
        sig = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        pass

    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))

    return kernel / np.sum(kernel)

def disk(k):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """
    ax = np.arange(-k // 2 + 1., k // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = (np.sqrt(pow(xx, 2) + pow(yy, 2)) <= (k - 1) / 2).astype(np.float32)

    return kernel

if __name__ == '__main__':
    rect_1 = np.array([[54.56637168141593, 91.65781710914455], [85.51592356687898, 90.92993630573248], [64.36440677966101, 123.17372881355932], [84.18115942028986, 109.3840579710145]])
    rect_2 = np.array([[64.0, 92.0, 1, 2], [86.0, 92.0, 1, 3], [64.0, 111.0, 1, 1], [86.0, 111.0, 1, 0]])
    print('IOU', calcIOU(rect_1, rect_2))
    pass
