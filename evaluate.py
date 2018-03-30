import numpy as np
import cv2
from QP import reconstructFloorplan, findMatches
from RecordReader import *
from train import *

# Disable
def blockPrint():
    return
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    return
    sys.stdout = sys.__stdout__

def evaluate(options):
    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass
    if os.path.exists(options.test_dir + '/dummy'):
        #os.rmdir(options.test_dir + '/dummy')
        pass
    else:
        os.mkdir(options.test_dir + '/dummy')
        pass

    if options.useCache == 2 and os.path.exists(options.test_dir + '/dummy/gt_dict.npy') and os.path.exists(options.test_dir + '/dummy/pred_dict.npy'):
        return

    if options.useCache == 1 and os.path.exists(options.test_dir + '/dummy/gt_dict.npy') and os.path.exists(options.test_dir + '/dummy/pred_dict.npy'):
        evaluateBatch(options)
        return

    if not os.path.exists(options.test_dir):
        os.system("mkdir -p %s"%options.test_dir)
        pass

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
    loss, loss_list = build_loss(options, pred_dict, gt_dict, dataset_flag, debug_dict, input_dict['flags'])

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

                gt = {'corner': gt['corner'], 'corner_values': gt['corner_values'], 'icon': gt['icon'], 'room': gt['room'], 'density': debug['x0_topdown'][:, :, :, -1], 'image_path': inp['image_path'], 'num_corners': gt['num_corners'], 'image_flags': image_flags}
                if iteration == 0:
                    gtAll = gt
                    predAll = pred
                else:
                    for k, v in gt.iteritems():
                        gtAll[k] = np.concatenate([gtAll[k], v], axis=0)
                        continue
                    for k, v in pred.iteritems():
                        predAll[k] = np.concatenate([predAll[k], v], axis=0)
                        continue
                    pass
                continue
        except tf.errors.OutOfRangeError:
            print('Finish testing')
            pass

        pass

    if options.useCache != -1:
        np.save(options.test_dir + '/dummy/gt_dict.npy', gtAll)
        np.save(options.test_dir + '/dummy/pred_dict.npy', predAll)
        pass
    if options.useCache == -2:
        return

    evaluateBatch(options, gtAll, predAll)
    return

def evaluateBatch(options, gt_dict=None, pred_dict=None):
    datasetFlag = 1
    if options.useCache != -1:
        if options.loss != '5':
            gt_dict = np.load(options.test_dir + '/dummy/gt_dict.npy')[()]
            pred_dict = np.load(options.test_dir + '/dummy/pred_dict.npy')[()]
        else:
            gt_dict = np.load(options.test_dir.replace('loss5', 'loss0') + '/dummy/gt_dict.npy')[()]
            pred_wc = np.load(options.test_dir.replace('loss5', 'loss0') + '/dummy/pred_dict.npy')[()]['corner'][:, :, :, :NUM_WALL_CORNERS]
            pred_oc = np.load(options.test_dir.replace('loss5', 'loss1') + '/dummy/pred_dict.npy')[()]['corner'][:, :, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4]
            pred_ic = np.load(options.test_dir.replace('loss5', 'loss2').replace('hybrid14', 'hybrid1') + '/dummy/pred_dict.npy')[()]['corner'][:, :, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8]
            pred_icon = np.load(options.test_dir.replace('loss5', 'loss3').replace('hybrid14', 'hybrid1') + '/dummy/pred_dict.npy')[()]['icon']
            pred_room = np.load(options.test_dir.replace('loss5', 'loss4').replace('hybrid14', 'hybrid1') + '/dummy/pred_dict.npy')[()]['room']
            pred_dict = {'corner': np.concatenate([pred_wc, pred_oc, pred_ic], axis=-1), 'icon': pred_icon, 'room': pred_room}
        pass

    if options.separateIconLoss:
        pred_icon_separate = softmax(np.load(options.test_dir.replace('wsf', 'wsf_loss3') + '/dummy/pred_dict.npy')[()]['icon'])
        pass
    #pred_dict['icon'] = np.load(options.test_dir.replace('wsf', 'wsf_loss3').replace('hybrid1', 'hybrid14').replace('dataset_1', '') + '/dummy/pred_dict.npy')[()]['icon']
    #pred_dict['corner'][:, :, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8] = np.load(options.test_dir.replace('wsf', 'wsf_loss2') + '/dummy/pred_dict.npy')[()]['corner'][:, :, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8]
    #pass

    if options.cornerLossType != 'mse':
        threshold = np.ones((HEIGHT, WIDTH, 1)) * 0.5
    else:
        threshold = np.ones((HEIGHT, WIDTH, 1)) * 0.5# HEATMAP_SCALE / 2
        pass

    statisticsSum = {k: [0.0, 0.0, 0.0] for k in ['wall', 'door', 'icon', 'room', 'neighbor', 'neighbor_all']}
    #print(pred_dict['corner'].max())
    pred_wc = pred_dict['corner'][:, :, :, :NUM_WALL_CORNERS]
    pred_oc = pred_dict['corner'][:, :, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4]
    pred_ic = pred_dict['corner'][:, :, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8]

    if options.branches != '5':
        pred_wc = sigmoid(pred_wc)
        pred_oc = sigmoid(pred_oc)
        pred_ic = sigmoid(pred_ic)
    else:
        threshold = np.ones((HEIGHT, WIDTH, 1)) * 0.3
        pass

    gt_wc = gt_dict['corner'][:, :, :, :NUM_WALL_CORNERS]
    gt_oc = gt_dict['corner'][:, :, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4]
    gt_ic = gt_dict['corner'][:, :, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8]

    names = []

    for batchIndex in xrange(gt_dict['corner'].shape[0]):
        #if batchIndex == 0:
        #continue

        #if options.branches == '4' and gt_dict['image_flags'][batchIndex] == 0:
        if options.evaluateImage and gt_dict['image_flags'][batchIndex] == 0:
            continue

        density = np.minimum(gt_dict['density'][batchIndex] * 255, 255).astype(np.uint8)
        density = np.stack([density, density, density], axis=2)

        pred_icon = softmax(pred_dict['icon'][batchIndex])
        pred_room = softmax(pred_dict['room'][batchIndex])
        if options.separateIconLoss:
            pred_icon[:, :, :-2] = pred_icon_separate[batchIndex][:, :, :-2]
            #pred_icon = pred_icon_separate[batchIndex]
            pass

        if False:
            #print('batch index', batchIndex)
            cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_density.png', density)
            #print('heatmap max value', pred_wc[batchIndex].max())

            if datasetFlag in [0, 1, 4]:
                cornerImage = drawSegmentationImage(np.concatenate([threshold, pred_wc[batchIndex]], axis=2), blackIndex=0)
                cornerImage[cornerImage == 0] = density[cornerImage == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_corner_pred.png', cornerImage)

                cornerImage = drawSegmentationImage(np.concatenate([threshold, gt_wc[batchIndex]], axis=2), blackIndex=0)
                cornerImage[cornerImage == 0] = density[cornerImage == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_corner_gt.png', cornerImage)
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
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_corner_heatmap.png', corner_rgb_img)
                pass

            if datasetFlag in [1, 4]:
                cornerImage = drawSegmentationImage(np.concatenate([threshold, pred_oc[batchIndex]], axis=2), blackIndex=0)
                cornerImage[cornerImage == 0] = density[cornerImage == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_opening_corner_pred.png', cornerImage)

                cornerImage = drawSegmentationImage(np.concatenate([threshold, pred_ic[batchIndex]], axis=2), blackIndex=0)
                cornerImage[cornerImage == 0] = density[cornerImage == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_icon_corner_pred.png', cornerImage)


                cornerImage = drawSegmentationImage(np.concatenate([threshold, gt_oc[batchIndex]], axis=2), blackIndex=0)
                cornerImage[cornerImage == 0] = density[cornerImage == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_opening_corner_gt.png', cornerImage)

                cornerImage = drawSegmentationImage(np.concatenate([threshold, gt_ic[batchIndex]], axis=2), blackIndex=0)
                cornerImage[cornerImage == 0] = density[cornerImage == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_icon_corner_gt.png', cornerImage)
                pass


            if datasetFlag in [1, 2, 3, 4]:
                icon_density = drawSegmentationImage(gt_dict['icon'][batchIndex], blackIndex=0)
                icon_density[icon_density == 0] = density[icon_density == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_icon_gt.png', icon_density)

                icon_density = drawSegmentationImage(pred_dict['icon'][batchIndex], blackIndex=0)
                icon_density[icon_density == 0] = density[icon_density == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_icon_pred.png', icon_density)
                pass

            if datasetFlag in [1, 3, 4]:
                room_density = drawSegmentationImage(gt_dict['room'][batchIndex], blackIndex=0)
                room_density[room_density == 0] = density[room_density == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_room_gt.png', room_density)

                room_density = drawSegmentationImage(pred_dict['room'][batchIndex], blackIndex=0)
                room_density[room_density == 0] = density[room_density == 0]
                cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_room_pred.png', room_density)
                pass


            if batchIndex == 0 and False:
                for c in xrange(22):
                    cv2.imwrite(options.test_dir + '/mask_' + str(c) + '.png', cv2.dilate(drawMaskImage(corner_segmentation[batchIndex] == c), np.ones((3, 3)), 3))
                    continue
                continue


        if batchIndex < options.visualizeReconstruction or True:
            if options.debug >= 0 and batchIndex != options.debug:
                continue
            names.append((batchIndex, gt_dict['image_path'][batchIndex]))
            print(batchIndex, 'start reconstruction', gt_dict['image_path'][batchIndex])
            if True:
                if options.debug == -1:
                    blockPrint()
                    pass

                # gtHeatmaps = gt_dict['corner'][batchIndex]
                #result_gt = reconstructFloorplan(gtHeatmaps[:, :, :NUM_WALL_CORNERS], gtHeatmaps[:, :, NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4], gtHeatmaps[:, :, NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8], segmentation2Heatmaps(gt_dict['icon'][batchIndex], NUM_ICONS), segmentation2Heatmaps(gt_dict['room'][batchIndex], NUM_ROOMS), density[:, :, 0], gt=True)
                orientationCorners = getOrientationCorners(gt_dict['corner_values'][batchIndex][:gt_dict['num_corners'][batchIndex]])
                result_gt = reconstructFloorplan(orientationCorners[:NUM_WALL_CORNERS], orientationCorners[NUM_WALL_CORNERS:NUM_WALL_CORNERS + 4], orientationCorners[NUM_WALL_CORNERS + 4:NUM_WALL_CORNERS + 8], segmentation2Heatmaps(gt_dict['icon'][batchIndex], NUM_ICONS), segmentation2Heatmaps(gt_dict['room'][batchIndex], NUM_ROOMS), density[:, :, 0], gt=True)

                #if batchIndex == 1:
                #exit(1)


                #pred_debug_dir = options.test_dir + '/' + str(batchIndex) + '_debug'
                pred_debug_dir = options.test_dir
                try:
                    os.mkdir(pred_debug_dir)
                    pass
                except OSError as e:
                    pass

                result_pred = reconstructFloorplan(pred_wc[batchIndex], pred_oc[batchIndex], pred_ic[batchIndex], pred_icon, pred_room, density[:, :, 0], gt_dict=result_gt, gt=False, debug_prefix=pred_debug_dir)

                if True:
                    try:
                        newWidth = newHeight = 1000
                        resizeResult(result_gt, newWidth, newHeight, WIDTH, HEIGHT)
                        resultImageGT = drawResultImageFinal(newWidth, newHeight, result_gt)
                        #cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_result_gt.png', resultImageGT)
                        cv2.imwrite(options.test_dir + '/' + gt_dict['image_path'][batchIndex] + '_gt.png', resultImageGT)

                        resizeResult(result_pred, newWidth, newHeight, WIDTH, HEIGHT)
                        resultImagePred = drawResultImageFinal(newWidth, newHeight, result_pred)
                        cv2.imwrite(options.test_dir + '/' + gt_dict['image_path'][batchIndex] + '_pred.png', resultImagePred)
                    except:
                        continue
                    continue

                if 'wall' not in result_pred or 'wall' not in result_gt:
                    print('invalid result')
                    continue

                statistics = findMatches(result_pred, result_gt, distanceThreshold=10)

                if options.drawFinal:
                    newWidth = newHeight = 1000
                    resizeResult(result_gt, newWidth, newHeight, WIDTH, HEIGHT)
                    resultImageGT = drawResultImageFinal(newWidth, newHeight, result_gt)
                    cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_result_gt.png', resultImageGT)

                    resizeResult(result_pred, newWidth, newHeight, WIDTH, HEIGHT)
                    resultImagePred = drawResultImageFinal(newWidth, newHeight, result_pred)
                    cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_result_pred.png', resultImagePred)

                    writeRepresentation('popup/data/floorplan_' + str(batchIndex) + '_gt.txt', newWidth, newHeight, result_gt)
                    writeRepresentation('popup/data/floorplan_' + str(batchIndex) + '_pred.txt', newWidth, newHeight, result_pred)
                    cv2.imwrite('popup/data/floorplan_' + str(batchIndex) + '_gt.png', resultImageGT)
                    cv2.imwrite('popup/data/floorplan_' + str(batchIndex) + '_pred.png', resultImagePred)
                    exit(1)
                else:
                    resultImage, iconImage = drawResultImage(WIDTH, HEIGHT, result_gt)
                    cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_reconstruction_wall_gt.png', resultImage)
                    iconImage[iconImage == 0] = density[iconImage == 0]
                    cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_reconstruction_icon_gt.png', iconImage)
                    resultImage, iconImage = drawResultImage(WIDTH, HEIGHT, result_pred)
                    cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_reconstruction_wall_pred.png', resultImage)
                    iconImage[iconImage == 0] = density[iconImage == 0]
                    cv2.imwrite(options.test_dir + '/' + str(batchIndex) + '_reconstruction_icon_pred.png', iconImage)
                    pass


                if options.debug == -1:
                    enablePrint()
                    pass
                if len(result_pred) == 0:
                    continue

                # print(result_pred)
                # print(result_pred['door'])
                # print('gt')
                # print(result_gt)
                # print(result_gt['door'])
                # exit(1)

                print('find predictions among ground-truths')
                #print(result_pred['wall'][2])
                #statistics = findMatches(result_pred, result_gt, distanceThreshold=10)
                #statistics = findMatches(result_gt, result_pred, distanceThreshold=10)

                #print('find ground-truths among predictions')
                #statistics = findMatches(result_gt, result_pred, distanceThreshold=10)
                #print(statistics)
                print('statistics', [(k, float(v[0]) / max(v[1], 1), float(v[0]) / max(v[2], 1)) for k, v in statistics.iteritems()])
                #print('topology statistics', [(k, float(v[0]) / max(v[1], 1), float(v[0]) / max(v[2], 1)) for k, v in topologyStatistics.iteritems()])
                print('finish reconstruction', gt_dict['image_path'][batchIndex])
                for k, v in statistics.iteritems():
                    if k in statisticsSum:
                        for c in xrange(3):
                            statisticsSum[k][c] += v[c]
                            continue
                    else:
                        print(k, 'not in', statisticsSum)
                    continue
                if options.debug >= 0:
                    exit(1)
                    pass
                pass

            # except Exception as e:
            #     #traceback.print_tb(e)
            #     print('exception-----------: ', e)
            #     #raise e
        continue
    print(names)
    print('final statistics', [(k, float(v[0]) / max(v[1], 1), float(v[0]) / max(v[2], 1)) for k, v in statisticsSum.iteritems()])
    np.save(options.test_dir + '/numbers.npy', statisticsSum)
    #print(statisticsSum)
    return
