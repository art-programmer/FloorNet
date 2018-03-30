import numpy as np
from skimage import measure
import cv2
import copy
from utils import *

NUM_WALL_CORNERS = 13
NUM_CORNERS = 21
CORNER_RANGES = {'wall': (0, 13), 'opening': (13, 17), 'icon': (17, 21)}
MAX_NUM_CORNERS = 300

NUM_FINAL_ICONS = 10
NUM_FINAL_ROOMS = 15
NUM_ICONS = 13
NUM_ROOMS = 16
HEIGHT=256
WIDTH=256
NUM_POINTS = 50000

NUM_INPUT_CHANNELS = 7
NUM_CHANNELS = [7, 64, 64, 64, 128, 256]
SIZES = [WIDTH, WIDTH // 2, WIDTH // 4, WIDTH // 8, WIDTH // 16, WIDTH // 32]



POINT_ORIENTATIONS = [[(2, ), (3, ), (0, ), (1, )], [(0, 3), (0, 1), (1, 2), (2, 3)], [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)], [(0, 1, 2, 3)]]


def getOrientationRanges(width, height):
    orientationRanges = [[width, 0, 0, 0], [width, height, width, 0], [width, height, 0, height], [0, height, 0, 0]]
    return orientationRanges

def getIconNames():
    iconNames = []
    iconLabelMap = getIconLabelMap()
    for iconName, _ in iconLabelMap.iteritems():
        iconNames.append(iconName)
        continue
    return iconNames

def getRoomLabelMap():
    labelMap = {}
    labelMap['living_room'] = 1
    labelMap['kitchen'] = 2
    labelMap['bedroom'] = 3
    labelMap['bathroom'] = 4
    labelMap['restroom'] = 4
    labelMap['office'] = 3
    labelMap['closet'] = 6
    labelMap['balcony'] = 7
    labelMap['corridor'] = 8
    labelMap['dining_room'] = 9
    labelMap['laundry_room'] = 10
    labelMap['garage'] = 11
    labelMap['recreation_room'] = 12
    labelMap['stairs'] = 13
    labelMap['other'] = 14
    labelMap['wall'] = 15
    return labelMap

def getLabelRoomMap():
    labelMap = {}
    labelMap[1] = 'living room'
    labelMap[2] = 'kitchen'
    labelMap[3] = 'bedroom'
    labelMap[4] = 'bathroom'
    labelMap[6] = 'closet'
    labelMap[7] = 'balcony'
    labelMap[8] = 'corridor'
    labelMap[9] = 'dining room'
    return labelMap

def getIconLabelMap():
    labelMap = {}
    labelMap['cooking_counter'] = 1
    labelMap['bathtub'] = 2
    labelMap['toilet'] = 3
    labelMap['washing_basin'] = 4
    labelMap['sofa'] = 5
    labelMap['cabinet'] = 6
    labelMap['bed'] = 7
    labelMap['table'] = 8
    labelMap['desk'] = 8
    labelMap['refrigerator'] = 9
    labelMap['TV'] = 0
    labelMap['entrance'] = 0
    labelMap['chair'] = 0
    labelMap['door'] = 11
    labelMap['window'] = 12
    return labelMap

def getLabelIconMap():
    labelMap = {}
    labelMap[1] = 'cooking_counter'
    labelMap[2] = 'bathtub'
    labelMap[3] = 'toilet'
    labelMap[4] = 'washing_basin'
    labelMap[5] = 'sofa'
    labelMap[6] = 'cabinet'
    labelMap[7] = 'bed'
    labelMap[8] = 'table'
    labelMap[9] = 'refrigerator'
    return labelMap

def getLabelMapNYU40():
    labelMap = {}
    labelMap[1] = 'wall'
    labelMap[2] = 'floor'
    labelMap[3] = 'cabinet'
    labelMap[4] = 'bed'
    labelMap[5] = 'chair'
    labelMap[6] = 'sofa'
    labelMap[7] = 'table'
    labelMap[8] = 'door'
    labelMap[9] = 'window'
    labelMap[10] = 'bookshelf'
    labelMap[11] = 'picture'
    labelMap[12] = 'cooking_counter'
    labelMap[13] = 'blinds'
    labelMap[14] = 'desk'
    labelMap[15] = 'shelf'
    labelMap[16] = 'curtain'
    labelMap[17] = 'dresser'
    labelMap[18] = 'pillow'
    labelMap[19] = 'mirror'
    labelMap[20] = 'entrance' #mat
    labelMap[21] = 'clothes'
    labelMap[22] = 'ceiling'
    labelMap[23] = 'book'
    labelMap[24] = 'refrigerator'
    labelMap[25] = 'TV'
    labelMap[26] = 'paper'
    labelMap[27] = 'towel'
    labelMap[28] = 'shower_curtain'
    labelMap[29] = 'box'
    labelMap[30] = 'whiteboard'
    labelMap[31] = 'person'
    labelMap[32] = 'nightstand'
    labelMap[33] = 'toilet'
    labelMap[34] = 'washing_basin'
    labelMap[35] = 'lamp'
    labelMap[36] = 'bathtub'
    labelMap[37] = 'bag'
    labelMap[38] = 'otherprop'
    labelMap[39] = 'otherstructure'
    labelMap[40] = 'unannotated'
    return labelMap

def getNYUScanNetMap():
    labelMap = np.zeros(41, dtype=np.int32)
    labelMap[1] = 1
    labelMap[2] = 2
    labelMap[3] = 19
    labelMap[4] = 6
    labelMap[5] = 3
    labelMap[6] = 8
    labelMap[7] = 4
    labelMap[8] = 14
    labelMap[9] = 15
    labelMap[10] = 7
    labelMap[11] = 18
    labelMap[12] = 13
    labelMap[13] = 12 #20 Blinds
    labelMap[14] = 5
    labelMap[15] = 7
    labelMap[16] = 12
    labelMap[17] = 19
    labelMap[18] = 20
    labelMap[19] = 20
    labelMap[20] = 20
    labelMap[21] = 20
    labelMap[22] = 1
    labelMap[23] = 20
    labelMap[24] = 17
    labelMap[25] = 20
    labelMap[26] = 20
    labelMap[27] = 20
    labelMap[28] = 16
    labelMap[29] = 20
    labelMap[30] = 20
    labelMap[31] = 20
    labelMap[32] = 20
    labelMap[33] = 11
    labelMap[34] = 9
    labelMap[35] = 20
    labelMap[36] = 10
    labelMap[37] = 20
    labelMap[38] = 20
    labelMap[39] = 20
    labelMap[40] = 0
    return labelMap

def getMatterportClassMap():
    classMap = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8,
        'i': 9,
        'j': 10,
        'k': 11,
        'l': 12,
        'm': 13,
        'n': 14,
        'o': 15,
        'p': 16,
        'r': 17,
        's': 18,
        't': 19,
        'u': 20,
        'v': 21,
        'w': 22,
        'x': 23,
        'y': 24,
        'z': 25,
        'B': 26,
        'C': 27,
        'D': 28,
        'S': 29,
        'Z': 30
    }
    return classMap

def calcLineDim(points, line):
  point_1 = points[line[0]]
  point_2 = points[line[1]]
  if abs(point_2[0] - point_1[0]) > abs(point_2[1] - point_1[1]):
    lineDim = 0
  else:
    lineDim = 1
  return lineDim

def calcLineDirection(line):
    return int(abs(line[0][0] - line[1][0]) < abs(line[0][1] - line[1][1]))

def calcLineDirectionPoints(points, line):
  point_1 = points[line[0]]
  point_2 = points[line[1]]
  if isinstance(point_1[0], tuple):
      point_1 = point_1[0]
      pass
  if isinstance(point_2[0], tuple):
      point_2 = point_2[0]
      pass
  return calcLineDirection((point_1, point_2))

def pointDistance(point_1, point_2):
    #return np.sqrt(pow(point_1[0] - point_2[0], 2) + pow(point_1[1] - point_2[1], 2))
    return max(abs(point_1[0] - point_2[0]), abs(point_1[1] - point_2[1]))

def sortLines(lines):
    newLines = []
    for line in lines:
        direction = calcLineDirection(line)
        if line[0][direction] < line[1][direction]:
            newLines.append((line[0], line[1]))
        else:
            newLines.append((line[1], line[0]))
            pass
        continue
    return newLines

def lineRange(line):
    direction = calcLineDirection(line)
    fixedValue = (line[0][1 - direction] + line[1][1 - direction]) / 2
    minValue = min(line[0][direction], line[1][direction])
    maxValue = max(line[0][direction], line[1][direction])
    return direction, fixedValue, minValue, maxValue

def findConnections(line_1, line_2, gap):
    connection_1 = -1
    connection_2 = -1
    pointConnected = False
    for c_1 in xrange(2):
        if pointConnected:
            break
        for c_2 in xrange(2):
            if pointDistance(line_1[c_1], line_2[c_2]) > gap:
                continue

            connection_1 = c_1
            connection_2 = c_2
            connectionPoint = ((line_1[c_1][0] + line_2[c_2][0]) / 2, (line_1[c_1][1] + line_2[c_2][1]) / 2)
            pointConnected = True
            break
        continue
    if pointConnected:
        return [connection_1, connection_2], connectionPoint
    direction_1, fixedValue_1, min_1, max_1 = lineRange(line_1)
    direction_2, fixedValue_2, min_2, max_2 = lineRange(line_2)
    if direction_1 == direction_2:
        return [-1, -1], (0, 0)

    #print(fixedValue_1, min_1, max_1, fixedValue_2, min_2, max_2)
    if min(fixedValue_1, max_2) < max(fixedValue_1, min_2) - gap or min(fixedValue_2, max_1) < max(fixedValue_2, min_1) - gap:
        return [-1, -1], (0, 0)

    if abs(min_1 - fixedValue_2) <= gap:
        return [0, 2], (fixedValue_2, fixedValue_1)
    if abs(max_1 - fixedValue_2) <= gap:
        return [1, 2], (fixedValue_2, fixedValue_1)
    if abs(min_2 - fixedValue_1) <= gap:
        return [2, 0], (fixedValue_2, fixedValue_1)
    if abs(max_2 - fixedValue_1) <= gap:
        return [2, 1], (fixedValue_2, fixedValue_1)
    return [2, 2], (fixedValue_2, fixedValue_1)

def lines2Corners(lines, gap, getSingularCorners=False):
    corners = []
    lineConnections = []
    for _ in xrange(len(lines)):
        lineConnections.append({})
        continue

    connectionCornerMap = {}
    connectionCornerMap[(1, 1)] = 4
    connectionCornerMap[(0, 1)] = 5
    connectionCornerMap[(0, 0)] = 6
    connectionCornerMap[(1, 0)] = 7
    connectionCornerMap[(2, 0)] = 8
    connectionCornerMap[(1, 2)] = 9
    connectionCornerMap[(2, 1)] = 10
    connectionCornerMap[(0, 2)] = 11
    connectionCornerMap[(2, 2)] = 12
    corners = []
    for lineIndex_1, line_1 in enumerate(lines):
        for lineIndex_2, line_2 in enumerate(lines):
            if lineIndex_2 == lineIndex_1:
                continue
            connections, connectionPoint = findConnections(line_1, line_2, gap=gap)
            if connections[0] == -1 and connections[1] == -1:
                continue
            if calcLineDirection(line_1) == calcLineDirection(line_2):
                print('overlap', line_1, line_2, connections)
                exit(1)
                pass
            if calcLineDirection(line_1) == 1:
                continue

            indices = [lineIndex_1, lineIndex_2]
            #print(lineIndex_1, lineIndex_2, connections)
            for c in xrange(2):
                if connections[c] in [0, 1] and connections[c] in lineConnections[indices[c]]:
                    print('duplicate corner', line_1, line_2, connections)
                    exit(1)
                    pass
                lineConnections[indices[c]][connections[c]] = True
                continue
            corners.append((connectionPoint, connectionCornerMap[tuple(connections)]))
            continue
        continue

    if getSingularCorners:
        singularCorners = []
        for lineIndex, connections in enumerate(lineConnections):
            if 0 not in connections:
                print('single corner', lines[lineIndex], connections)
                singularCorners.append((lineIndex, 0))
                pass
            if 1 not in connections:
                print('single corner', lines[lineIndex], connections)
                singularCorners.append((lineIndex, 1))
                pass
            continue
        return corners, singularCorners

    return corners

def drawWallMask(walls, width, height, thickness=3, indexed=False):
    if indexed:
        wallMask = np.full((height, width), -1, dtype=np.int32)
        for wallIndex, wall in enumerate(walls):
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=wallIndex, thickness=thickness)
            continue
    else:
        wallMask = np.zeros((height, width), dtype=np.int32)
        for wall in walls:
            cv2.line(wallMask, (int(wall[0][0]), int(wall[0][1])), (int(wall[1][0]), int(wall[1][1])), color=1, thickness=thickness)
            continue
        wallMask = wallMask.astype(np.bool)
        pass
    return wallMask

def mergeLines(line_1, line_2):
    direction_1, fixedValue_1, min_1, max_1 = lineRange(line_1)
    direction_2, fixedValue_2, min_2, max_2 = lineRange(line_2)
    fixedValue = (fixedValue_1 + fixedValue_2) / 2
    if direction_1 == 0:
        return [[min(min_1, min_2), fixedValue], [max(max_1, max_2), fixedValue]]
    else:
        return [[fixedValue, min(min_1, min_2)], [fixedValue, max(max_1, max_2)]]
    return

def findIntersection(line_1, line_2):
    direction_1, fixedValue_1, min_1, max_1 = lineRange(line_1)
    direction_2, fixedValue_2, min_2, max_2 = lineRange(line_2)
    if direction_1 == 0:
        return (fixedValue_2, fixedValue_1)
    else:
        return (fixedValue_1, fixedValue_2)
    return

def extendLine(line, point):
    direction, fixedValue, min_value, max_value = lineRange(line)
    if direction == 0:
        return ((min(min_value, point[direction]), fixedValue), (max(max_value, point[direction]), fixedValue))
    else:
        return ((fixedValue, min(min_value, point[direction])), (fixedValue, max(max_value, point[direction])))
    return

def divideWalls(walls):
    horizontalWalls = []
    verticalWalls = []
    for wall in walls:
        if calcLineDirection(wall) == 0:
            horizontalWalls.append(wall)
        else:
            verticalWalls.append(wall)
            pass
        continue
    return horizontalWalls, verticalWalls


def connectWalls(walls, roomSegmentation, gap=3):
    width = roomSegmentation.shape[1]
    height = roomSegmentation.shape[0]
    roomBoundary = np.zeros(roomSegmentation.shape, dtype=np.bool)
    for direction in xrange(2):
        for shift in [-1, 1]:
            roomBoundary = np.logical_or(roomBoundary, roomSegmentation != np.roll(roomSegmentation, shift, axis=direction))
            continue
        continue
    roomBoundary = roomBoundary.astype(np.uint8)
    roomBoundary[0] = roomBoundary[-1] = roomBoundary[:, 0] = roomBoundary[:, -1] = 0

    uncoveredBoundary = roomBoundary.copy()
    wallGroups = divideWalls(walls)
    wallMasks = [drawWallMask(walls, width, height, indexed=True, thickness=gap * 2) for walls in wallGroups]

    uncoveredBoundary[wallMasks[0] >= 0] = 0
    uncoveredBoundary[wallMasks[1] >= 0] = 0
    uncoveredBoundary = cv2.dilate(uncoveredBoundary, np.ones((3, 3)), iterations=gap)
    components = measure.label(uncoveredBoundary, background=0)

    connectedWalls = []

    for walls, wallMask in zip(wallGroups, wallMasks):

        newWalls = copy.deepcopy(walls)
        invalidWallIndices = []
        for label in xrange(components.min() + 1, components.max() + 1):
            mask = components == label
            wallIndices = np.unique(wallMask[mask]).tolist()
            if -1 in wallIndices:
                wallIndices.remove(-1)
                pass
            if len(wallIndices) != 2:
                continue

            wall_1 = newWalls[wallIndices[0]]
            wall_2 = newWalls[wallIndices[1]]

            direction_1, fixedValue_1, min_1, max_1 = lineRange(wall_1)
            direction_2, fixedValue_2, min_2, max_2 = lineRange(wall_2)

            if direction_1 == direction_2:
                if abs(fixedValue_1 - fixedValue_2) < gap:
                    newWallIndex = len(newWalls)
                    wallMask[wallMask == wallIndices[0]] = newWallIndex
                    wallMask[wallMask == wallIndices[1]] = newWallIndex
                    newWall = mergeLines(wall_1, wall_2)
                    newWalls.append(newWall)
                    invalidWallIndices.append(wallIndices[0])
                    invalidWallIndices.append(wallIndices[1])
                    pass
                pass
            #     else:
            #         print(wall_1, wall_2)
            #         ys, xs = mask.nonzero()
            #         newWall = [[xs.min(), ys.min()], [xs.max(), ys.max()]]
            #         newWallDirection = calcLineDirection(newWall)
            #         if newWallDirection != direction_1 and newWall[1][1 - newWallDirection] - newWall[0][1 - newWallDirection] < gap * 2 + 1:
            #             fixedValue = (newWall[1][1 - newWallDirection] + newWall[0][1 - newWallDirection]) / 2
            #             newWall[1][1 - newWallDirection] = newWall[0][1 - newWallDirection] = fixedValue
            #             newWalls.append(newWall)
            #             pass
            #         pass
            # else:
            #     assert(False)
            #     intersectionPoint = findIntersection(wall_1, wall_2)
            #     newWalls[wallIndices[0]] = extendLine(wall_1, intersectionPoint)
            #     newWalls[wallIndices[1]] = extendLine(wall_2, intersectionPoint)
            #     pass
            continue

        #print(invalidWallIndices)
        invalidWallIndices = sorted(invalidWallIndices, key=lambda x: -x)
        for index in invalidWallIndices:
            del newWalls[index]
            continue
        connectedWalls += newWalls
        continue

    newWalls = connectedWalls
    wallMask = drawWallMask(newWalls, width, height, indexed=True, thickness=gap * 2)
    uncoveredBoundary = roomBoundary.copy()
    uncoveredBoundary[wallMask >= 0] = 0
    uncoveredBoundary = cv2.dilate(uncoveredBoundary, np.ones((3, 3)), iterations=gap)
    components = measure.label(uncoveredBoundary, background=0)

    #cv2.imwrite('test/segmentation.png', drawSegmentationImage(components))

    for label in xrange(components.min() + 1, components.max() + 1):
        mask = components == label
        #cv2.imwrite('test/mask_' + str(label) + '.png', drawMaskImage(mask))
        wallIndices = np.unique(wallMask[mask]).tolist()
        if -1 in wallIndices:
            wallIndices.remove(-1)
            pass

        lines = [newWalls[index] for index in wallIndices]
        #cv2.imwrite('test/mask_' + str(label) + '_segment.png', drawMaskImage(mask))
        #cv2.imwrite('test/mask_' + str(label) + '.png', drawMaskImage(drawWallMask(lines, width, height)))

        horizontalLines, verticalLines = divideWalls(lines)
        if len(horizontalLines) > 0 and len(verticalLines) > 0:
            continue
        #print(label, wallIndices, len(horizontalLines), len(verticalLines))
        for direction, lines in enumerate([horizontalLines, verticalLines]):
            if len(lines) < 2:
                continue
            #wall_1 = lines[0]
            #wall_2 = lines[1]
            #print(wall_1, wall_2)
            #direction_1, fixedValue_1, min_1, max_1 = lineRange(wall_1)
            #direction_2, fixedValue_2, min_2, max_2 = lineRange(wall_2)
            #values = [line[direction] for line in lines]

            #print(wall_1, wall_2)
            ys, xs = mask.nonzero()
            newWall = [[xs.min(), ys.min()], [xs.max(), ys.max()]]
            newWallDirection = calcLineDirection(newWall)
            #print(label, wallIndices, newWallDirection, direction, newWall[1][1 - newWallDirection] - newWall[0][1 - newWallDirection])
            if newWallDirection != direction and newWall[1][1 - newWallDirection] - newWall[0][1 - newWallDirection] <= (gap * 2 + 2) * 2:
                fixedValue = (newWall[1][1 - newWallDirection] + newWall[0][1 - newWallDirection]) / 2
                newWall[1][1 - newWallDirection] = newWall[0][1 - newWallDirection] = fixedValue
                values = [line[0][newWallDirection] for line in lines] + [line[1][newWallDirection] for line in lines]
                min_value = min(values)
                max_value = max(values)
                newWall[0][newWallDirection] = min_value
                newWall[1][newWallDirection] = max_value

                newWalls.append(newWall)
                #print('new orthogonal wall', newWall)
                pass

            continue
        continue

    wallMask = drawWallMask(newWalls, width, height, indexed=True, thickness=gap * 2)
    uncoveredBoundary = roomBoundary.copy()
    uncoveredBoundary[wallMask >= 0] = 0
    uncoveredBoundary = cv2.dilate(uncoveredBoundary, np.ones((3, 3)), iterations=gap)
    components = measure.label(uncoveredBoundary, background=0)

    for label in xrange(components.min() + 1, components.max() + 1):
        mask = components == label
        wallIndices = np.unique(wallMask[mask]).tolist()
        if -1 in wallIndices:
            wallIndices.remove(-1)
            pass
        if len(wallIndices) != 2:
            continue

        wall_1 = newWalls[wallIndices[0]]
        wall_2 = newWalls[wallIndices[1]]

        #print(wall_1, wall_2)

        direction_1 = calcLineDirection(wall_1)
        direction_2 = calcLineDirection(wall_2)

        if direction_1 != direction_2:
            intersectionPoint = findIntersection(wall_1, wall_2)
            newWalls[wallIndices[0]] = extendLine(wall_1, intersectionPoint)
            newWalls[wallIndices[1]] = extendLine(wall_2, intersectionPoint)
            pass
        continue

    # try:
    #     _, singularCorners = lines2Corners(newWalls, gap=gap, getSingularCorners=True)
    #     for _, singularCorner_1 in enumerate(singularCorners):
    #         for singularCorner_2 in singularCorners[_ + 1:]:
    #             wall_1 = newWalls[singularCorner_1[0]]
    #             wall_2 = newWalls[singularCorner_2[0]]
    #             corner_1 = wall_1[singularCorner_1[1]]
    #             corner_2 = wall_2[singularCorner_2[1]]
    #             if pointDistance(corner_1, corner_2) < (gap * 2 + 1) * 2:
    #                 intersectionPoint = findIntersection(wall_1, wall_2)
    #                 newWalls[singularCorner_1[0]] = extendLine(wall_1, intersectionPoint)
    #                 newWalls[singularCorner_2[0]] = extendLine(wall_2, intersectionPoint)
    #                 pass
    #             continue
    #         continue
    # except:
    #     pass

    return newWalls

def extractLines(lineMask, lengthThreshold=11, widthThreshold=5):
    lines = []
    components = measure.label(lineMask, background=0)
    for label in xrange(components.min() + 1, components.max() + 1):
        mask = components == label
        ys, xs = mask.nonzero()
        line = [[xs.min(), ys.min()], [xs.max(), ys.max()]]
        direction = calcLineDirection(line)
        if abs(line[1][1 - direction] - line[0][1 - direction]) > widthThreshold or abs(line[1][direction] - line[0][direction]) < lengthThreshold:
            continue
        fixedValue = (line[1][1 - direction] + line[0][1 - direction]) / 2
        line[1][1 - direction] = line[0][1 - direction] = fixedValue
        lines.append(line)
        continue
    return lines


def drawPoints(filename, width, height, points, backgroundImage=None, pointSize=5, pointColor=None):
  colorMap = ColorPalette(NUM_CORNERS).getColorMap()
  if np.all(np.equal(backgroundImage, None)):
    image = np.zeros((height, width, 3), np.uint8)
  else:
    if backgroundImage.ndim == 2:
      image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 3])
    else:
      image = backgroundImage
      pass
  pass
  no_point_color = pointColor is None
  for point in points:
    if no_point_color:
        pointColor = colorMap[point[2] * 4 + point[3]]
        pass
    #print('used', pointColor)
    #print('color', point[2] , point[3])
    image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width)] = pointColor
    continue

  if filename != '':
    cv2.imwrite(filename, image)
    return
  else:
    return image

def drawPointsSeparately(path, width, height, points, backgroundImage=None, pointSize=5):
  if np.all(np.equal(backgroundImage, None)):
    image = np.zeros((height, width, 13), np.uint8)
  else:
    image = np.tile(np.expand_dims(backgroundImage, -1), [1, 1, 13])
    pass

  for point in points:
    image[max(int(round(point[1])) - pointSize, 0):min(int(round(point[1])) + pointSize, height), max(int(round(point[0])) - pointSize, 0):min(int(round(point[0])) + pointSize, width), int(point[2] * 4 + point[3])] = 255
    continue
  for channel in xrange(13):
    cv2.imwrite(path + '_' + str(channel) + '.png', image[:, :, channel])
    continue
  return

def drawLineMask(width, height, points, lines, lineWidth = 5, backgroundImage = None):
  lineMask = np.zeros((height, width))

  for lineIndex, line in enumerate(lines):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    direction = calcLineDirectionPoints(points, line)

    fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
    minValue = int(min(point_1[direction], point_2[direction]))
    maxValue = int(max(point_1[direction], point_2[direction]))
    if direction == 0:
      lineMask[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1] = 1
    else:
      lineMask[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width)] = 1
      pass
    continue
  return lineMask



def drawLines(filename, width, height, points, lines, lineLabels = [], backgroundImage = None, lineWidth = 5, lineColor = None):
  colorMap = ColorPalette(len(lines)).getColorMap()
  if backgroundImage is None:
    image = np.ones((height, width, 3), np.uint8) * 0
  else:
    if backgroundImage.ndim == 2:
      image = np.stack([backgroundImage, backgroundImage, backgroundImage], axis=2)
    else:
      image = backgroundImage
      pass
    pass

  for lineIndex, line in enumerate(lines):
    point_1 = points[line[0]]
    point_2 = points[line[1]]
    direction = calcLineDirectionPoints(points, line)


    fixedValue = int(round((point_1[1 - direction] + point_2[1 - direction]) / 2))
    minValue = int(round(min(point_1[direction], point_2[direction])))
    maxValue = int(round(max(point_1[direction], point_2[direction])))
    if len(lineLabels) == 0:
      if np.any(lineColor == None):
        lineColor = np.random.rand(3) * 255
        pass
      if direction == 0:
        image[max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue + 1, :] = lineColor
      else:
        image[minValue:maxValue + 1, max(fixedValue - lineWidth, 0):min(fixedValue + lineWidth + 1, width), :] = lineColor
    else:
      labels = lineLabels[lineIndex]
      isExterior = False
      if direction == 0:
        for c in xrange(3):
          image[max(fixedValue - lineWidth, 0):min(fixedValue, height), minValue:maxValue, c] = colorMap[labels[0]][c]
          image[max(fixedValue, 0):min(fixedValue + lineWidth + 1, height), minValue:maxValue, c] = colorMap[labels[1]][c]
          continue
      else:
        for c in xrange(3):
          image[minValue:maxValue, max(fixedValue - lineWidth, 0):min(fixedValue, width), c] = colorMap[labels[1]][c]
          image[minValue:maxValue, max(fixedValue, 0):min(fixedValue + lineWidth + 1, width), c] = colorMap[labels[0]][c]
          continue
        pass
      pass
    continue

  if filename == '':
    return image
  else:
    cv2.imwrite(filename, image)


def drawRectangles(filename, width, height, points, rectangles, labels, lineWidth = 2, backgroundImage = None, rectangleColor = None):
  colorMap = ColorPalette(NUM_ICONS).getColorMap()
  if backgroundImage is None:
    image = np.ones((height, width, 3), np.uint8) * 0
  else:
    image = backgroundImage
    pass

  for rectangleIndex, rectangle in enumerate(rectangles):
    point_1 = points[rectangle[0]]
    point_2 = points[rectangle[1]]
    point_3 = points[rectangle[2]]
    point_4 = points[rectangle[3]]


    if len(labels) == 0:
      if rectangleColor is None:
        color = np.random.rand(3) * 255
      else:
        color = rectangleColor
    else:
      color = colorMap[labels[rectangleIndex]]
      pass

    x_1 = int(round((point_1[0] + point_3[0]) / 2))
    x_2 = int(round((point_2[0] + point_4[0]) / 2))
    y_1 = int(round((point_1[1] + point_2[1]) / 2))
    y_2 = int(round((point_3[1] + point_4[1]) / 2))

    cv2.rectangle(image, (x_1, y_1), (x_2, y_2), color=tuple(color.tolist()), thickness = 2)

    # point_1 = (int(point_1[0]), int(point_1[1]))
    # point_2 = (int(point_2[0]), int(point_2[1]))
    # point_3 = (int(point_3[0]), int(point_3[1]))
    # point_4 = (int(point_4[0]), int(point_4[1]))

    # image[max(point_1[1] - lineWidth, 0):min(point_1[1] + lineWidth, height), point_1[0]:point_2[0] + 1, :] = color
    # image[max(point_3[1] - lineWidth, 0):min(point_3[1] + lineWidth, height), point_3[0]:point_4[0] + 1, :] = color
    # image[point_1[1]:point_3[1] + 1, max(point_1[0] - lineWidth, 0):min(point_1[0] + lineWidth, width), :] = color
    # image[point_2[1]:point_4[1] + 1, max(point_2[0] - lineWidth, 0):min(point_2[0] + lineWidth, width), :] = color

    continue

  if filename == '':
    return image
  else:
    cv2.imwrite(filename, image)
    pass


def drawResultImage(width, height, result):
    resultImage = drawLines('', width, height, result['wall'][0], result['wall'][1], result['wall'][2], None, lineWidth=3)
    resultImage = drawLines('', width, height, result['door'][0], result['door'][1], [], resultImage, lineWidth=2, lineColor=0)
    iconImage = drawRectangles('', width, height, result['icon'][0], result['icon'][1], result['icon'][2], lineWidth=2)
    return resultImage, iconImage

def resizeResult(result, width, height, oriWidth=256, oriHeight=256):
    result['wall'][0] = [[float(point[0]) / oriWidth * width, float(point[1]) / oriHeight * height, point[2], point[3]] for point in result['wall'][0]]
    result['door'][0] = [[float(point[0]) / oriWidth * width, float(point[1]) / oriHeight * height, point[2], point[3]] for point in result['door'][0]]
    result['icon'][0] = [[float(point[0]) / oriWidth * width, float(point[1]) / oriHeight * height, point[2], point[3]] for point in result['icon'][0]]
    #result['room'][0] = [(cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8) for mask in result['room'][0]]
    return

def drawResultImageFinal(width, height, result):
    colorMap = np.array([(224, 255, 192), (255, 160, 96), (255, 224, 128), (192, 255, 255), (192, 255, 255), (192, 255, 255), (192, 192, 224), (224, 255, 192), (255, 224, 224), (224, 224, 224)])
    borderColorMap = np.array([(128, 192, 64), (192, 64, 64), (192, 128, 64), (0, 128, 192), (0, 128, 192), (0, 128, 192), (128, 64, 160), (128, 192, 64), (192, 64, 0), (255, 255, 255)])
    colorMap = np.concatenate([np.full(shape=(1, 3), fill_value=0), colorMap, borderColorMap], axis=0).astype(np.uint8)
    colorMap = colorMap[:, ::-1]

    labelRoomMap = getLabelRoomMap()

    roomSegmentation = np.zeros((height, width), dtype=np.int32)

    roomsInfo = []
    wall_dict = result['wall']
    wallMask = drawWallMask([(wall_dict[0][line[0]], wall_dict[0][line[1]]) for line in wall_dict[1]], width, height, thickness=3)
    roomRegions = measure.label(1 - wallMask, background=0)
    #cv2.imwrite('test/' + str(dictIndex) + '_segmentation_regions.png', drawSegmentationImage(roomRegions))
    backgroundIndex = roomRegions.min()
    wallPoints = wall_dict[0]
    roomLabels = {}
    sizes = np.array([width, height])
    for wallIndex, wallLabels in enumerate(wall_dict[2]):
        wallLine = wall_dict[1][wallIndex]
        lineDim = calcLineDim(wallPoints, wallLine)
        #print('wall', wallIndex, wallPoints[wallLine[0]][:2], wallPoints[wallLine[1]][:2])
        center = np.round((np.array(wallPoints[wallLine[0]][:2]) + np.array(wallPoints[wallLine[1]][:2])) / 2).astype(np.int32)

        for c in xrange(2):
            direction = c * 2 - 1
            if lineDim == 1:
                direction *= -1
                pass
            point = center
            for offset in xrange(10):
                point[1 - lineDim] += direction
                if point[lineDim] < 0 or point[lineDim] >= sizes[lineDim]:
                    break
                roomIndex = roomRegions[point[1], point[0]]
                if roomIndex != backgroundIndex:
                    #print(roomIndex, wallLabels[c], wallLabels, point.tolist())
                    #mask = roomRegions == roomIndex
                    #mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=1)
                    #roomSegmentation[mask] = wallLabels[c]
                    #rooms[wallLabels[c]].append(cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=wallLineWidth))
                    #roomRegions[mask] = backgroundIndex
                    if roomIndex not in roomLabels:
                        roomLabels[roomIndex] = {}
                        pass
                    roomLabels[roomIndex][wallLabels[c]] = True
                    break
                continue
            continue
        continue


    rooms = []
    indexMap = {}
    for roomIndex, labels in roomLabels.iteritems():
        #print(roomIndex, labels)
        if roomIndex == roomRegions[0][0]:
            continue
        indexMap[roomIndex] = len(rooms)
        mask = roomRegions == roomIndex
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3)), iterations=3)

        # if 7 in labels and 2 not in labels:
        #     labels[2] = True
        #     pass
        # if 5 in labels and 3 not in labels:
        #     labels[3] = True
        #     pass
        # if 9 in labels and 1 not in labels:
        #     labels[1] = True
        #     pass
        rooms.append((mask, labels))
        continue

    wallLineWidth = 5
    # foregroundMask = roomSegmentation > 0
    # foregroundMask = cv2.dilate(foregroundMask, np.ones((3, 3)), iterations=wallLineWidth)
    # roomSegmentation[foregroundMask] =
    for mask, labels in rooms:
        label = min([label for label in labels])
        if label < 0:
            continue
        kernel = np.zeros((3, 3))
        kernel[1:, 1:] = 1
        #mask = cv2.erode(mask.astype(np.uint8), kernel.astype(np.uint8), iterations=1)
        erodedMask = cv2.erode(mask, np.ones((3, 3)), iterations=wallLineWidth)
        roomSegmentation[mask.astype(np.bool)] = label + 10
        roomSegmentation[erodedMask.astype(np.bool)] = label

        continue
    image = colorMap[roomSegmentation.reshape(-1)].reshape((height, width, 3))

    pointColor = tuple((np.array([0.3, 0.3, 0.9]) * 255).astype(np.uint8).tolist())
    for wallLine in result['wall'][1]:
        for pointIndex in wallLine:
            point = result['wall'][0][pointIndex]
            cv2.circle(image, (int(point[0]), int(point[1])), color=pointColor, radius=8, thickness=-1)
            cv2.circle(image, (int(point[0]), int(point[1])), color=(255, 255, 255), radius=4, thickness=-1)
            continue
        continue

    lineSegmentLength = 20.0
    for doorLine in result['door'][1]:
        point_1 = np.array(result['door'][0][doorLine[0]][:2]).astype(np.float32)
        point_2 = np.array(result['door'][0][doorLine[1]][:2]).astype(np.float32)
        lineDim = calcLineDim(result['door'][0], doorLine)
        for i in xrange(int(abs(point_1[lineDim] - point_2[lineDim]) / lineSegmentLength + 1)):
            ratio = i * lineSegmentLength / abs(point_1[lineDim] - point_2[lineDim])
            if ratio >= 1:
                break
            startPoint = point_1 + ratio * (point_2 - point_1)
            ratio = (i + 0.5) * lineSegmentLength / abs(point_1[lineDim] - point_2[lineDim])
            ratio = min(ratio, 1)
            endPoint = point_1 + ratio * (point_2 - point_1)
            cv2.line(image, (startPoint[0], startPoint[1]), (endPoint[0], endPoint[1]), color=(0, 0, 0), thickness=4)
            continue
        for point in [point_1, point_2]:
            startPoint = point.copy()
            startPoint[1 - lineDim] += lineSegmentLength / 2
            endPoint = point.copy()
            endPoint[1 - lineDim] -= lineSegmentLength / 2
            cv2.line(image, (startPoint[0], startPoint[1]), (endPoint[0], endPoint[1]), color=(0, 0, 0), thickness=2)
            continue
        continue



    labelIconMap = getLabelIconMap()
    iconPos = []
    for iconIndex, (icon, label) in enumerate(zip(result['icon'][1], result['icon'][2])):
        name = labelIconMap[label + 1]
        iconImage = cv2.imread('icons/' + name + '.jpg')

        points = [result['icon'][0][pointIndex] for pointIndex in icon]
        x_1 = int(round((points[0][0] + points[2][0]) / 2))
        x_2 = int(round((points[1][0] + points[3][0]) / 2))
        y_1 = int(round((points[0][1] + points[1][1]) / 2))
        y_2 = int(round((points[2][1] + points[3][1]) / 2))

        iconSize = iconImage.shape #(y, x)
        #print('icon_size', iconSize)
        icon_is_landscape = iconSize[1] >  iconSize[0]

        slot_size = (x_2 - x_1 + 1, y_2 - y_1 + 1)
        slot_center = np.array((x_1 + slot_size[0]/2, y_1 + slot_size[1] / 2))
        slot_is_landscape = slot_size[0] > slot_size[1]

        min_dist = float('inf')
        line = None
        close_line_dim = 0
        for wallIndex, wallLabels in enumerate(wall_dict[2]):
            wallLine = wall_dict[1][wallIndex]
            lineDim = calcLineDim(wallPoints, wallLine)
            center = np.round((np.array(wallPoints[wallLine[0]][:2]) + np.array(wallPoints[wallLine[1]][:2])) / 2).astype(np.int32)
            point1=np.array(wallPoints[wallLine[0]][:2])
            point2 = np.array(wallPoints[wallLine[1]][:2])
            n = point2 - point1
            dist = np.dot((point1 - slot_center) - np.dot((point1 - slot_center), n), n)
            #print('indices', wallIndex, wallLabels, wallLine)
            #       print('points', wallPoints[wallLine[0]], wallPoints[wallLine[1]])
            #       pass

            if dist < 5:
                min_dist = dist
                line = (point1, point2)
                close_line_dim = lineDim
                pass
            pass

        #sys.stderr.write("{}, {}, {}, {}, {}\n".format(y_1, y_2, x_1, x_2, iconImage.shape))
        print('has line: ', line, name, close_line_dim)
        if name == "toilet":
            if line is not None:
                if close_line_dim == 0: #x
                    y_pos = (line[0][1] + line[1][1]) / 2
                    if y_pos > y_2: #toilet is below
                        print('first case rot')
                        iconImage = rotateImage(iconImage, 2)
                    elif y_pos < y_1: # toilet is above
                        pass # do nothing
                    else:
                        print("bad case", x_1, x_2, y_1, y_2, line)
                        pass
                else: #y
                    x_pos = (line[0][0] + line[1][0])/2
                    print('here', x_pos, x_1, x_2)
                    if x_pos > x_2: #toilet is to the left
                        pass # do nothing
                    elif x_pos < x_1: # toilet is to the right
                        print(slot_is_landscape, icon_is_landscape)
                        if slot_is_landscape:
                            iconImage = rotateImage(iconImage, 2)
                        pass # do nothing
                    else:
                        print("bad case", x_1, x_2, y_1, y_2, line)
                        pass
                pass
        elif name == "washing_basin":
            if line is not None:
                if close_line_dim == 0: #x

                    y_pos = (line[0][1] + line[1][1]) / 2
                    print(y_pos, y_1, y_2, 'y')
                    if y_pos > y_2: #toilet is below
                        iconImage = rotateImage(iconImage, 2)
                        pass
                    elif y_pos < y_1: # toilet is above
                        pass # do nothing
                    else:
                        print("bad case", x_1, x_2, y_1, y_2, line)
                        pass
                else: #y
                    x_pos = (line[0][0] + line[1][0])/2
                    print(x_pos, x_1, x_2 , 'x')
                    if x_pos > x_2: #toilet is to the left
                        pass # do nothing
                    elif x_pos < x_1: # toilet is to the right
                        if not slot_is_landscape:
                            iconImage = rotateImage(iconImage, 2)
                            pass # do nothing
                        pass
                    else:
                        print("bad case", x_1, x_2, y_1, y_2, line)
                        pass
                    pass
                pass
            pass
        pass
        if slot_is_landscape != icon_is_landscape:
            iconImage = rotateImage(iconImage, 1)


        iconImage = cv2.resize(iconImage, slot_size)


        image[y_1:y_2 + 1, x_1:x_2 + 1] = iconImage
        if name == "washing_basin":
            print('basin pose', [x_1, y_1, x_2, y_2])
        iconPos.append([x_1,y_1, x_2, y_2])
        continue

    fontSize = 0.7
    for mask, labels in rooms:
        label = min([label for label in labels])
        if label <= 0:
            continue
        ys, xs = mask.nonzero()
        print(xs.mean(), ys.mean(), label)

        #label_half_size_x = int(fontSize * len(labelRoomMap[label]) / 2 * 20)
        #label_half_size_y = int(fontSize / 2 * 20)
        ret, baseline = cv2.getTextSize(labelRoomMap[label], fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontSize, thickness=1)
        print(labelRoomMap[label])
        #print('ret', ret)
        center = findBestTextLabelCenter(iconPos, xs, ys, ret[0]/2, ret[1]/2)
        print('comp', [xs.mean(), ys.mean()], center)
        #center = np.round([xs.mean(), ys.mean()]).astype(np.int32)
        if center is not None:
            cv2.putText(image, labelRoomMap[label], (center[0] - ret[0]/2, center[1] + ret[1]/2), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontSize, color=(0, 0, 0), thickness=1)
        else:
            if label != 4:
                import sys
                sys.stderr.write("panic! I cannot find valid position to put label in room: {}, {}\n".format(label, labelRoomMap[label]))
        continue
    print('end draw')
    #cv2.imwrite('test/result.png', image)
    #exit(1)
    #cv2.imwrite('test/region.png', drawSegmentationImage(roomRegions))
    # for regionIndex in xrange(roomRegions.max() + 1):
    #     cv2.imwrite('test/mask_' + str(regionIndex) + '.png', drawMaskImage(roomRegions == regionIndex))
    #     continue
    #resultImage = drawLines('', width, height, result['wall'][0], result['wall'][1], result['wall'][2], None, lineWidth=3)
    #resultImage = drawLines('', width, height, result['door'][0], result['door'][1], [], resultImage, lineWidth=2, lineColor=0)
    #iconImage = drawRectangles('', width, height, result['icon'][0], result['icon'][1], result['icon'][2], lineWidth=2)
    return image
def findBestTextLabelCenter( icon_pos, xs, ys, label_half_size_x, label_half_size_y):
    center = np.array([xs.mean(), ys.mean()])
    icon_pos = np.array(icon_pos)
    room_points = np.array([xs, ys]).transpose()
    min_point = room_points.min(axis=0, keepdims=True)
    max_point = room_points.max(axis=0, keepdims=True)
    size = np.array([label_half_size_x, label_half_size_y])
    print('size', size)
    avail_min_point = min_point + size
    avail_max_point = max_point - size

    avail_points = np.logical_and(room_points > avail_min_point, room_points < avail_max_point)
    avail_points = np.all(avail_points, axis=1)

    room_points_aug = np.tile(room_points[:, :, np.newaxis], [1, 1, icon_pos.shape[0]])

    room_points_gt_tl_x = room_points_aug[:, 0, :] > icon_pos[:, 0] - size[0] - 5
    room_points_lt_br_x = room_points_aug[:, 0, :] < icon_pos[:, 2] + size[0] + 5

    room_points_gt_tl_y = room_points_aug[:, 1, :] > icon_pos[:, 1] - size[1] - 5
    room_points_lt_br_y = room_points_aug[:, 1, :] < icon_pos[:, 3] + size[1] + 5

    room_points_in_square = np.logical_and(room_points_gt_tl_x, room_points_lt_br_x)
    room_points_in_square = np.logical_and(room_points_in_square, room_points_gt_tl_y)
    room_points_in_square = np.logical_and(room_points_in_square, room_points_lt_br_y)

    #room_points_in_square = np.all(room_points_in_square, axis=1)
    room_points_in_square = np.any(room_points_in_square, axis=1)

    room_points_not_in_square = np.logical_not(room_points_in_square)

    good_points_mask = np.logical_and(avail_points, room_points_not_in_square)
    good_points = room_points[good_points_mask]
    good_points_center_dist = np.linalg.norm(good_points - center, axis=1)
    #good_points_center_dist = np.sum(np.abs(good_points - center), axis=1)
    #print('icon_pos')
    #print(icon_pos)
    #print('goodpoints')
    #print(center)
    #print(good_points)
    #print(good_points_center_dist)
    if len(good_points) == 0:
        #print('give up')
        return None
        #return np.round(center).astype(np.int32)
    best_point_idx = np.argmin(good_points_center_dist, axis=0)
    #print('cost', good_points_center_dist[best_point_idx])
    #print('best points')
    #print(good_points[best_point_idx])
    return good_points[best_point_idx]

def rotateImage(image, orientation):
    if orientation == 0:
        return image
    elif orientation == 1:
        return np.flip(image.transpose((1, 0, 2)), axis=0)
    elif orientation == 2:
        return np.flip(np.flip(image, axis=1), axis=0)
    else:
        return np.flip(image.transpose(1, 0, 2), axis=1)
    return

def projectIndices(pointIndices, pointSegmentation, min_x, max_x, min_y, max_y):
    if max_x - min_x == 1 and max_y - min_y == 1:
        pointIndices[pointSegmentation[:, 2]] = min_y * WIDTH + min_x
        return
    elif max_x - min_x >= max_y - min_y:
        middle_x = int((max_x + min_x + 1) / 2)
        mask_1 = pointSegmentation[:, 0] < middle_x
        projectIndices(pointIndices, pointSegmentation[mask_1], min_x, middle_x, min_y, max_y)
        mask_2 = pointSegmentation[:, 0] >= middle_x
        projectIndices(pointIndices, pointSegmentation[mask_2], middle_x, max_x, min_y, max_y)
    else:
        middle_y = int((max_y + min_y + 1) / 2)
        mask_1 = pointSegmentation[:, 1] < middle_y
        projectIndices(pointIndices, pointSegmentation[mask_1], min_x, max_x, min_y, middle_y)
        mask_2 = pointSegmentation[:, 1] >= middle_y
        projectIndices(pointIndices, pointSegmentation[mask_2], min_x, max_x, middle_y, max_y)
        pass
    return

def drawCornerSegmentation(corners, radius=1, width=WIDTH, height=HEIGHT):
    cornerSegmentation = np.zeros((height, width), dtype=np.int64)
    for corner in corners:
        cornerSegmentation[max(corner[1] - radius + 1, 0):min(corner[1] + radius, height - 1), max(corner[0] - radius + 1, 0):min(corner[0] + radius, width - 1)] = corner[2]
        continue
    return cornerSegmentation

def getOrientationCorners(corners, cornerSize=3):
    orientationCorners = [[] for _ in xrange(NUM_CORNERS)]
    for corner in corners:
        orientationCorners[corner[2] - 1].append(((corner[0], corner[1]), (corner[0] - cornerSize, corner[1] - cornerSize), (corner[0] + cornerSize, corner[1] + cornerSize)))
        continue
    return orientationCorners

def getGTPrimitives(gt_dict):
    result_dict = {'wall': [wallPoints, filteredWallLines, filteredWallLabels], 'door': [doorPoints, filteredDoorLines, []], 'icon': [iconPoints, filteredIcons, filteredIconTypes]}
    return

def writeRepresentation(filename, width, height, result_dict):
    labelMap = [11, 1, 2, 3, 4, 3, 6, 7, 8, 2,]
    labelIconMap = getLabelIconMap()
    with open(filename, 'w') as f:
        f.write(str(width) + '\t' + str(height) + '\n')
        f.write(str(len(result_dict['wall'][1])) + '\n')
        for wallLine, wallLabels in zip(result_dict['wall'][1], result_dict['wall'][2]):
            point_1 = result_dict['wall'][0][wallLine[0]]
            point_2 = result_dict['wall'][0][wallLine[1]]
            lineDim = calcLineDim(result_dict['wall'][0], wallLine)
            if point_1[lineDim] > point_2[lineDim]:
                point_1[lineDim], point_2[lineDim] = point_2[lineDim], point_1[lineDim]
                pass
            f.write(str(int(point_1[0])) + '\t' + str(int(point_1[1])) + '\t' + str(int(point_2[0])) + '\t' + str(int(point_2[1])) + '\t' + str(labelMap[wallLabels[0]]) + '\t' + str(labelMap[wallLabels[1]]) + '\n')
            continue
        for doorLine in result_dict['door'][1]:
            point_1 = result_dict['door'][0][doorLine[0]]
            point_2 = result_dict['door'][0][doorLine[1]]
            lineDim = calcLineDim(result_dict['door'][0], doorLine)
            if point_1[lineDim] > point_2[lineDim]:
                point_1[lineDim], point_2[lineDim] = point_2[lineDim], point_1[lineDim]
                pass
            f.write(str(int(point_1[0])) + '\t' + str(int(point_1[1])) + '\t' + str(int(point_2[0])) + '\t' + str(int(point_2[1])) + '\tdoor\t1\t1\n')
            continue
        #print(len(result_dict['icon'][1]))
        for icon, iconLabel in zip(result_dict['icon'][1], result_dict['icon'][2]):
            #print(iconLabel, labelIconMap[iconLabel + 1])
            points = np.array([result_dict['icon'][0][pointIndex][:2] for pointIndex in icon]).astype(np.int32)
            mins = points.min(0)
            maxs = points.max(0)
            f.write(str(int(mins[0])) + '\t' + str(int(mins[1])) + '\t' + str(int(maxs[0])) + '\t' + str(int(maxs[1])) + '\t' + labelIconMap[iconLabel + 1] + '\t1\t1\n')
            continue
        f.close()
        pass
    return
