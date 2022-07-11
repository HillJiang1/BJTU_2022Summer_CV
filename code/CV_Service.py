from keras.preprocessing.image import img_to_array
from keras.models import load_model
import subprocess
from oldcare.track import CentroidTracker
from oldcare.track import TrackableObject
from imutils.video import FPS
import imutils
import shutil
import dlib
from oldcare.utils import fileassistant
import argparse
from oldcare.facial import FaceUtil
from oldcare.audio import audioplayer
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import time

def fallDetection():
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=False, default='',
                    help="")
    args = vars(ap.parse_args())
    input_video = args['filename']

    # 控制陌生人检测
    fall_timing = 0  # 计时开始
    fall_start_time = 0  # 开始时间
    fall_limit_time = 1  # if >= 1 seconds, then he/she falls.

    # 全局变量
    model_path = '../models/fall_detection.hdf5'
    output_fall_path = '../supervision/fall'
    # your python path
    python_path = '/Users/hilljiang/opt/anaconda3/envs/CV/bin/python'

    # 全局常量
    TARGET_WIDTH = 64
    TARGET_HEIGHT = 64

    # 初始化摄像头
    if not input_video:
        vs = cv2.VideoCapture(0)
        time.sleep(2)
    else:
        vs = cv2.VideoCapture(input_video)

    # 加载模型
    model = load_model(model_path)

    print('[INFO] 开始检测是否有人摔倒...')
    # 不断循环
    counter = 0
    while True:
        counter += 1
        # grab the current frame
        (grabbed, image) = vs.read()

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if input_video and not grabbed:
            break

        if not input_video:
            image = cv2.flip(image, 1)

        roi = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # determine facial expression
        (fall, normal) = model.predict(roi)[0]
        label = "Fall (%.2f)" % (fall) if fall > normal else "Normal (%.2f)" % (normal)

        # display the label and bounding box rectangle on the output frame
        cv2.putText(image, label, (image.shape[1] - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        if fall > normal:
            if fall_timing == 0:  # just start timing
                fall_timing = 1
                fall_start_time = time.time()
            else:  # alredy started timing
                fall_end_time = time.time()
                difference = fall_end_time - fall_start_time

                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(time.time()))

                if difference < fall_limit_time:
                    print('[INFO] %s, 走廊, 摔倒仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                else:  # strangers appear
                    event_desc = '有人摔倒!!!'
                    event_location = '走廊'
                    print('[EVENT] %s, 走廊, 有人摔倒!!!' % (current_time))
                    cv2.imwrite(os.path.join(output_fall_path,
                                             'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), image)  # snapshot
                    # insert into database
                    command = '%s inserting.py --event_desc %s --event_type 3 --event_location %s' % (
                    python_path, event_desc, event_location)
                    p = subprocess.Popen(command, shell=True)

        cv2.imshow('Fall detection', image)

        # Press 'ESC' for exiting video
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    vs.release()
    cv2.destroyAllWindows()

def fenceDetection():
    # 得到当前时间
    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                 time.localtime(time.time()))
    print('[INFO] %s 禁止区域检测程序启动了.' % (current_time))

    # 传入参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=False, default='',
                    help="")
    args = vars(ap.parse_args())

    # 全局变量
    prototxt_file_path = '../models/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
    # Contains the Caffe deep learning model files.
    # We’ll be using a MobileNet Single Shot Detector (SSD),
    # “Single Shot Detectors for object detection”.
    model_file_path = '../models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
    output_fence_path = '../supervision/fence'
    input_video = args['filename']
    skip_frames = 30  # of skip frames between detections
    # your python path
    python_path = '/Users/hilljiang/opt/anaconda3/envs/CV/bin/python'

    # 超参数
    # minimum probability to filter weak detections
    minimum_confidence = 0.80

    # 物体识别模型能识别的物体（21种）
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike",
               "person", "pottedplant", "sheep", "sofa", "train",
               "tvmonitor"]

    # if a video path was not supplied, grab a reference to the webcam
    if not input_video:
        print("[INFO] starting video stream...")
        vs = cv2.VideoCapture(0)
        time.sleep(2)
    else:
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(input_video)

    # 加载物体识别模型
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_file_path, model_file_path)

    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = None
    H = None

    # instantiate our centroid tracker, then initialize a list to store
    # each of our dlib correlation trackers, followed by a dictionary to
    # map each unique object ID to a TrackableObject
    ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0

    # start the frames per second throughput estimator
    fps = FPS().start()

    # loop over frames from the video stream
    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        ret, frame = vs.read()

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if input_video and not ret:
            break

        if not input_video:
            frame = cv2.flip(frame, 1)

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if totalFrames % skip_frames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
            net.setInput(blob)
            detections = net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > minimum_confidence:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(rgb)
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # draw a rectangle around the people
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        event_desc = '有人闯入禁止区域!!!'
                        event_location = '院子'
                        print('[EVENT] %s, 院子, 有人闯入禁止区域!!!' % (current_time))
                        cv2.imwrite(
                            os.path.join(output_fence_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                            frame)  # snapshot

                        # insert into database
                        command = '%s inserting.py --event_desc %s --event_type 4 --event_location %s' % (
                        python_path, event_desc, event_location)
                        p = subprocess.Popen(command, shell=True)

                    # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4,
                       (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            # ("Up", totalUp),
            ("Down", totalDown),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Prohibited Area", frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        totalFrames += 1
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))  # 14.19
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))  # 90.43

    # close any open windows
    vs.release()
    cv2.destroyAllWindows()

def facialExpressionAndStrangerDetection():
    # 得到当前时间
    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                 time.localtime(time.time()))
    print('[INFO] %s 陌生人检测程序和表情检测程序启动了.' % (current_time))

    # 传入参数
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--filename", required=False, default='',
                    help="")
    args = vars(ap.parse_args())
    input_video = args['filename']

    # 全局变量
    facial_recognition_model_path = '../models/face_recognition_hog.pickle'
    facial_expression_model_path = '../models/face_expression_miniVGG.hdf5'

    output_stranger_path = '../supervision/strangers'
    output_smile_path = '../supervision/smile'

    people_info_path = '../info/people_info.csv'
    facial_expression_info_path = '../info/facial_expression_info.csv'
    # your python path
    python_path = '/Users/hilljiang/opt/anaconda3/envs/CV/bin/python'

    # 全局常量
    FACIAL_EXPRESSION_TARGET_WIDTH = 28
    FACIAL_EXPRESSION_TARGET_HEIGHT = 28

    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 480

    ANGLE = 20

    # 得到 ID->姓名的map 、 ID->职位类型的map、
    # 摄像头ID->摄像头名字的map、表情ID->表情名字的map
    id_card_to_name, id_card_to_type = fileassistant.get_people_info(
        people_info_path)
    facial_expression_id_to_name = fileassistant.get_facial_expression_info(
        facial_expression_info_path)

    # 控制陌生人检测
    strangers_timing = 0  # 计时开始
    strangers_start_time = 0  # 开始时间
    strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.

    # 控制微笑检测
    facial_expression_timing = 0  # 计时开始
    facial_expression_start_time = 0  # 开始时间
    facial_expression_limit_time = 2  # if >= 2 seconds, he/she is smiling

    # 初始化摄像头
    if not input_video:
        vs = cv2.VideoCapture(0)
        time.sleep(2)
    else:
        vs = cv2.VideoCapture(input_video)

    # 初始化人脸识别模型
    faceutil = FaceUtil(facial_recognition_model_path)
    facial_expression_model = load_model(facial_expression_model_path)

    print('[INFO] 开始检测陌生人和表情...')
    # 不断循环
    counter = 0
    while True:
        counter += 1
        # grab the current frame
        (grabbed, frame) = vs.read()

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if input_video and not grabbed:
            break

        if not input_video:
            frame = cv2.flip(frame, 1)

        frame = imutils.resize(frame, width=VIDEO_WIDTH,
                               height=VIDEO_HEIGHT)  # 压缩，加快识别速度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

        face_location_list, names = faceutil.get_face_location_and_name(
            frame)

        # 得到画面的四分之一位置和四分之三位置，并垂直划线
        one_fourth_image_center = (int(VIDEO_WIDTH / 4),
                                   int(VIDEO_HEIGHT / 4))
        three_fourth_image_center = (int(VIDEO_WIDTH / 4 * 3),
                                     int(VIDEO_HEIGHT / 4 * 3))

        cv2.line(frame, (one_fourth_image_center[0], 0),
                 (one_fourth_image_center[0], VIDEO_HEIGHT),
                 (0, 255, 255), 1)
        cv2.line(frame, (three_fourth_image_center[0], 0),
                 (three_fourth_image_center[0], VIDEO_HEIGHT),
                 (0, 255, 255), 1)

        # 处理每一张识别到的人脸
        for ((left, top, right, bottom), name) in zip(face_location_list,
                                                      names):

            # 将人脸框出来
            rectangle_color = (0, 0, 255)
            if id_card_to_type[name] == 'old_people':
                rectangle_color = (0, 0, 128)
            elif id_card_to_type[name] == 'employee':
                rectangle_color = (255, 0, 0)
            elif id_card_to_type[name] == 'volunteer':
                rectangle_color = (0, 255, 0)
            else:
                pass
            cv2.rectangle(frame, (left, top), (right, bottom),
                          rectangle_color, 2)

            # 陌生人检测逻辑
            if 'Unknown' in names:  # alert
                if strangers_timing == 0:  # just start timing
                    strangers_timing = 1
                    strangers_start_time = time.time()
                else:  # already started timing
                    strangers_end_time = time.time()
                    difference = strangers_end_time - strangers_start_time

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))

                    if difference < strangers_limit_time:
                        print('[INFO] %s, 房间, 陌生人仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                    else:  # strangers appear
                        event_desc = '陌生人出现!!!'
                        event_location = '房间'
                        print('[EVENT] %s, 房间, 陌生人出现!!!' % (current_time))
                        cv2.imwrite(os.path.join(output_stranger_path,
                                                 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                    frame)  # snapshot

                        # insert into database
                        command = '%s inserting.py --event_desc %s --event_type 2 --event_location %s' % (
                        python_path, event_desc, event_location)
                        p = subprocess.Popen(command, shell=True)

                        # 开始陌生人追踪
                        unknown_face_center = (int((right + left) / 2),
                                               int((top + bottom) / 2))

                        cv2.circle(frame, (unknown_face_center[0],
                                           unknown_face_center[1]), 4, (0, 255, 0), -1)

                        direction = ''
                        # face locates too left, servo need to turn right,
                        # so that face turn right as well
                        if unknown_face_center[0] < one_fourth_image_center[0]:
                            direction = 'right'
                        elif unknown_face_center[0] > three_fourth_image_center[0]:
                            direction = 'left'

                        # adjust to servo
                        if direction:
                            print('%d-摄像头需要 turn %s %d 度' % (counter,
                                                             direction, ANGLE))

            else:  # everything is ok
                strangers_timing = 0

            # 表情检测逻辑
            # 如果不是陌生人，且对象是老人
            if name != 'Unknown' and id_card_to_type[name] == 'old_people':
                # 表情检测逻辑
                roi = gray[top:bottom, left:right]
                roi = cv2.resize(roi, (FACIAL_EXPRESSION_TARGET_WIDTH,
                                       FACIAL_EXPRESSION_TARGET_HEIGHT))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # determine facial expression
                (neural, smile) = facial_expression_model.predict(roi)[0]
                facial_expression_label = 'Neural' if neural > smile else 'Smile'

                if facial_expression_label == 'Smile':  # alert
                    if facial_expression_timing == 0:  # just start timing
                        facial_expression_timing = 1
                        facial_expression_start_time = time.time()
                    else:  # already started timing
                        facial_expression_end_time = time.time()
                        difference = facial_expression_end_time - facial_expression_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        if difference < facial_expression_limit_time:
                            print(
                                '[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (current_time, id_card_to_name[name], difference))
                        else:  # he/she is really smiling
                            event_desc = '%s正在笑' % (id_card_to_name[name])
                            event_location = '房间'
                            print('[EVENT] %s, 房间, %s正在笑.' % (current_time, id_card_to_name[name]))
                            cv2.imwrite(os.path.join(output_smile_path,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                        frame)  # snapshot

                            # insert into database
                            command = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d' % (
                            python_path, event_desc, event_location, int(name))
                            p = subprocess.Popen(command, shell=True)

                else:  # everything is ok
                    facial_expression_timing = 0

            else:  # 如果是陌生人，则不检测表情
                facial_expression_label = ''

            # 人脸识别和表情识别都结束后，把表情和人名写上
            # (同时处理中文显示问题)
            img_PIL = Image.fromarray(cv2.cvtColor(frame,
                                                   cv2.COLOR_BGR2RGB))

            draw = ImageDraw.Draw(img_PIL)
            final_label = id_card_to_name[name] + ': ' + facial_expression_id_to_name[
                facial_expression_label] if facial_expression_label else id_card_to_name[name]
            draw.text((left, top - 30), final_label,
                      font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf', 40),
                      fill=(255, 0, 0))  # linux

            # 转换回OpenCV格式
            frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        # show our detected faces along with smiling/not smiling labels
        cv2.imshow("Checking Strangers and Ole People's Face Expression",
                   frame)

        # Press 'ESC' for exiting video
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    # cleanup the camera and close any open windows
    vs.release()
    cv2.destroyAllWindows()

def faceCollect(imagedir,id):
    audio_dir = '../audios'

    # 控制参数
    error = 0
    start_time = None
    limit_time = 2  # 2 秒

    action_list = ['blink', 'open_mouth', 'smile', 'rise_head', 'bow_head',
                   'look_left', 'look_right']
    action_map = {'blink': '请眨眼', 'open_mouth': '请张嘴',
                  'smile': '请笑一笑', 'rise_head': '请抬头',
                  'bow_head': '请低头', 'look_left': '请看左边',
                  'look_right': '请看右边'}
    # 设置摄像头
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    faceutil = FaceUtil()

    counter = 0
    while True:
        counter += 1
        _, image = cam.read()
        if counter <= 10:  # 放弃前10帧
            continue
        image = cv2.flip(image, 1)

        if error == 1:
            end_time = time.time()
            difference = end_time - start_time
            print(difference)
            if difference >= limit_time:
                error = 0

        face_location_list = faceutil.get_face_location(image)
        for (left, top, right, bottom) in face_location_list:
            cv2.rectangle(image, (left, top), (right, bottom),
                          (0, 0, 255), 2)

        cv2.imshow('Collecting Faces', image)  # show the image
        # Press 'ESC' for exiting video
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

        face_count = len(face_location_list)
        if error == 0 and face_count == 0:  # 没有检测到人脸
            print('[WARNING] 没有检测到人脸')
            audioplayer.play_audio(os.path.join(audio_dir,
                                                'no_face_detected.mp3'))
            error = 1
            start_time = time.time()
        elif error == 0 and face_count == 1:  # 可以开始采集图像了
            print('[INFO] 可以开始采集图像了')
            audioplayer.play_audio(os.path.join(audio_dir,
                                                'start_image_capturing.mp3'))
            break
        elif error == 0 and face_count > 1:  # 检测到多张人脸
            print('[WARNING] 检测到多张人脸')
            audioplayer.play_audio(os.path.join(audio_dir,
                                                'multi_faces_detected.mp3'))
            error = 1
            start_time = time.time()
        else:
            pass

    # 新建目录
    if os.path.exists(os.path.join(imagedir, id)):
        print(imagedir + "路径已经存在")
    else:
        os.mkdir(os.path.join(imagedir, id))

    # 开始采集人脸
    for action in action_list:
        audioplayer.play_audio(os.path.join(audio_dir, action + '.mp3'))
        action_name = action_map[action]

        counter = 1
        for i in range(15):
            print('%s-%d' % (action_name, i))
            _, img_OpenCV = cam.read()
            img_OpenCV = cv2.flip(img_OpenCV, 1)
            origin_img = img_OpenCV.copy()  # 保存时使用

            face_location_list = faceutil.get_face_location(img_OpenCV)
            for (left, top, right, bottom) in face_location_list:
                cv2.rectangle(img_OpenCV, (left, top),
                              (right, bottom), (0, 0, 255), 2)

            img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV,
                                                   cv2.COLOR_BGR2RGB))

            draw = ImageDraw.Draw(img_PIL)
            draw.text((int(image.shape[1] / 2), 30), action_name,
                      font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf', 40),
                      fill=(255, 0, 0))  # linux

            # 转换回OpenCV格式
            img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),
                                      cv2.COLOR_RGB2BGR)

            cv2.imshow('Collecting Faces', img_OpenCV)  # show the image

            image_name = os.path.join(imagedir,id,
                                      action + '_' + str(counter) + '.jpg')
            cv2.imwrite(image_name, origin_img)
            # Press 'ESC' for exiting video
            k = cv2.waitKey(100) & 0xff
            if k == 27:
                break
            counter += 1

    # 结束
    print('[INFO] 采集完毕')
    audioplayer.play_audio(os.path.join(audio_dir, 'end_capturing.mp3'))

    # 释放全部资源
    cam.release()
    cv2.destroyAllWindows()
