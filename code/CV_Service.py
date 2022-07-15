# from keras.preprocessing.image import img_to_array
from keras.models import load_model
import subprocess
from keras_preprocessing.image import img_to_array
from oldcare.track import CentroidTracker
from oldcare.track import TrackableObject
from imutils.video import FPS
import imutils
import shutil
import mediapipe as mp
import joblib
import dlib
from oldcare.utils import fileassistant
import argparse
from oldcare.facial import FaceUtil
from oldcare.conv.ResNet import resnet_18
from oldcare.audio import audioplayer
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import os
import time
import sklearn
from scipy.spatial import distance as dist


class cv:
    def __init__(self,id):
        if(id==1):#volunteer
            # 全局变量
            self.pixel_per_metric = None
            self.input_video = '../images/tests/volunteer.mov'
            self.output_activity_path = '../supervision/activity'
            self.model_path = '../models/face_recognition_hog.pickle'
            self.people_info_path = '../info/people_info.csv'
            self.camera_turned = 0
            self.python_path = '/Users/hilljiang/opt/anaconda3/envs/CV/bin/python'  # your python path
            self.sTime = 0
            self.interval = 0
            # 全局常量
            self.FACE_ACTUAL_WIDTH = 20  # 单位厘米   姑且认为所有人的脸都是相同大小
            self.VIDEO_WIDTH = 640
            self.VIDEO_HEIGHT = 480
            self.ANGLE = 20
            self.ACTUAL_DISTANCE_LIMIT = 100  # cm
            self.save_activity = '../../BJTU_2022Summer_Back_End/static/activity'
            # 得到 ID->姓名的map 、 ID->职位类型的map
            self.id_card_to_name, self.id_card_to_type = fileassistant.get_people_info(self.people_info_path)

            # 初始化摄像头
            if not self.input_video:
                self.vs = cv2.VideoCapture(0)
                time.sleep(2)
            else:
                self.vs = cv2.VideoCapture(self.input_video)

            # 加载模型
            self.faceutil = FaceUtil(self.model_path)
            self.count = 0
            self.eTime = 0
            self.sTime = 0
            self.interval = 0
            print('[INFO] 开始检测义工和老人是否有互动...')
            # 不断循环
            self.counter = 0
        if(id==2):#fall
            self.pose_knn = joblib.load('../models/PoseKeypoint.joblib')
            self.output_fall_path = '../supervision/fall'
            self.python_path = '/Users/hilljiang/opt/anaconda3/envs/CV/bin/python'  # your python path
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.mp_pose = mp.solutions.pose
            self.prevTime = 0
            self.save_fall = '../../BJTU_2022Summer_Back_End/static/fall'
            self.keyXYZ = [
                "nose_x",
                "nose_y",
                "nose_z",
                "left_eye_inner_x",
                "left_eye_inner_y",
                "left_eye_inner_z",
                "left_eye_x",
                "left_eye_y",
                "left_eye_z",
                "left_eye_outer_x",
                "left_eye_outer_y",
                "left_eye_outer_z",
                "right_eye_inner_x",
                "right_eye_inner_y",
                "right_eye_inner_z",
                "right_eye_x",
                "right_eye_y",
                "right_eye_z",
                "right_eye_outer_x",
                "right_eye_outer_y",
                "right_eye_outer_z",
                "left_ear_x",
                "left_ear_y",
                "left_ear_z",
                "right_ear_x",
                "right_ear_y",
                "right_ear_z",
                "mouth_left_x",
                "mouth_left_y",
                "mouth_left_z",
                "mouth_right_x",
                "mouth_right_y",
                "mouth_right_z",
                "left_shoulder_x",
                "left_shoulder_y",
                "left_shoulder_z",
                "right_shoulder_x",
                "right_shoulder_y",
                "right_shoulder_z",
                "left_elbow_x",
                "left_elbow_y",
                "left_elbow_z",
                "right_elbow_x",
                "right_elbow_y",
                "right_elbow_z",
                "left_wrist_x",
                "left_wrist_y",
                "left_wrist_z",
                "right_wrist_x",
                "right_wrist_y",
                "right_wrist_z",
                "left_pinky_x",
                "left_pinky_y",
                "left_pinky_z",
                "right_pinky_x",
                "right_pinky_y",
                "right_pinky_z",
                "left_index_x",
                "left_index_y",
                "left_index_z",
                "right_index_x",
                "right_index_y",
                "right_index_z",
                "left_thumb_x",
                "left_thumb_y",
                "left_thumb_z",
                "right_thumb_x",
                "right_thumb_y",
                "right_thumb_z",
                "left_hip_x",
                "left_hip_y",
                "left_hip_z",
                "right_hip_x",
                "right_hip_y",
                "right_hip_z",
                "left_knee_x",
                "left_knee_y",
                "left_knee_z",
                "right_knee_x",
                "right_knee_y",
                "right_knee_z",
                "left_ankle_x",
                "left_ankle_y",
                "left_ankle_z",
                "right_ankle_x",
                "right_ankle_y",
                "right_ankle_z",
                "left_heel_x",
                "left_heel_y",
                "left_heel_z",
                "right_heel_x",
                "right_heel_y",
                "right_heel_z",
                "left_foot_index_x",
                "left_foot_index_y",
                "left_foot_index_z",
                "right_foot_index_x",
                "right_foot_index_y",
                "right_foot_index_z"
            ]
            print(len(self.keyXYZ))
            self.count = 0
            self.sTime = 0
            self.eTime = 0
            self.interval = 0
            self.res_point = []
            self.cap = cv2.VideoCapture("../images/tests/Fall_Trim.mp4")
        if(id==3):#fence
            self.current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                         time.localtime(time.time()))
            print('[INFO] %s 禁止区域检测程序启动了.' % (self.current_time))

            # 传入参数
            # ap = argparse.ArgumentParser()
            # ap.add_argument("-f", "--filename", required=False, default='',
            #                 help="")
            # args = vars(ap.parse_args())

            # 全局变量
            self.prototxt_file_path = '../models/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
            # Contains the Caffe deep learning model files.
            # We’ll be using a MobileNet Single Shot Detector (SSD),
            # “Single Shot Detectors for object detection”.
            self.model_file_path = '../models/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
            self.output_fence_path = '../supervision/fence'
            self.input_video = "../images/tests/fence.mov"
            # self.input_video = None
            self.skip_frames = 30  # of skip frames between detections
            # your python path
            self.python_path = '/Users/hilljiang/opt/anaconda3/envs/CV/bin/python'
            self.count = 0
            self.sTime = 0
            self.interval = 0
            self.eTime = 0
            self.save_fence = '../../BJTU_2022Summer_Back_End/static/fence'
            # 超参数
            # minimum probability to filter weak detections
            self.minimum_confidence = 0.80

            # 物体识别模型能识别的物体（21种）
            self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                       "bottle", "bus", "car", "cat", "chair",
                       "cow", "diningtable", "dog", "horse", "motorbike",
                       "person", "pottedplant", "sheep", "sofa", "train",
                       "tvmonitor"]

            # if a video path was not supplied, grab a reference to the webcam
            if not self.input_video:
                print("[INFO] starting video stream...")
                self.vs = cv2.VideoCapture(0)
                time.sleep(2)
            else:
                print("[INFO] opening video file...")
                # self.vs = cv2.VideoCapture(self.input_video)
                print('right')
                self.vs = cv2.VideoCapture("rtsp://admin:admin@192.168.31.199:8554/live")

            # 加载物体识别模型
            print("[INFO] loading model...")
            self.net = cv2.dnn.readNetFromCaffe(self.prototxt_file_path, self.model_file_path)

            # initialize the frame dimensions (we'll set them as soon as we read
            # the first frame from the video)
            self.W = None
            self.H = None

            # instantiate our centroid tracker, then initialize a list to store
            # each of our dlib correlation trackers, followed by a dictionary to
            # map each unique object ID to a TrackableObject
            self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
            self.trackers = []
            self.trackableObjects = {}

            # initialize the total number of frames processed thus far, along
            # with the total number of objects that have moved either up or down
            self.totalFrames = 0
            self.totalDown = 0
            self.totalUp = 0

            # start the frames per second throughput estimator
            self.fps = FPS().start()
        if(id==4):#room
            self.current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                         time.localtime(time.time()))
            print('[INFO] %s 陌生人检测程序和表情检测程序启动了.' % (self.current_time))

            # # 传入参数
            # ap = argparse.ArgumentParser()
            # ap.add_argument("-f", "--filename", required=False, default='',
            #                 help="")
            # args = vars(ap.parse_args())
            self.input_video = '../images/tests/emotions.mov'

            # 全局变量
            self.facial_recognition_model_path = '../models/face_recognition_hog.pickle'
            self.facial_expression_model_path = '../models/face_expression_multiMood.hdf5'

            self.output_stranger_path = '../supervision/strangers'
            self.output_smile_path = '../supervision/smile'

            self.people_info_path = '../info/people_info.csv'
            self.facial_expression_info_path = '../info/facial_expression_info.csv'
            # your python path
            self.python_path = '/Users/hilljiang/opt/anaconda3/envs/CV/bin/python'  # your python path

            # 全局常量
            self.FACIAL_EXPRESSION_TARGET_WIDTH = 28
            self.FACIAL_EXPRESSION_TARGET_HEIGHT = 28

            self.VIDEO_WIDTH = 640
            self.VIDEO_HEIGHT = 480

            self.ANGLE = 20

            # 得到 ID->姓名的map 、 ID->职位类型的map、
            # 摄像头ID->摄像头名字的map、表情ID->表情名字的map
            self.id_card_to_name, self.id_card_to_type = fileassistant.get_people_info(
                self.people_info_path)
            self.facial_expression_id_to_name = fileassistant.get_facial_expression_info(
                self.facial_expression_info_path)

            # 控制陌生人检测
            self.strangers_timing = 0  # 计时开始
            self.strangers_start_time = 0  # 开始时间
            self.strangers_limit_time = 2  # if >= 2 seconds, then he/she is a stranger.
            self.count1 = 0
            self.sTime = 0
            self.interval = 0
            self.eTime = 0
            self.save_stranger = '../../BJTU_2022Summer_Back_End/static/stranger'
            # 控制微笑检测
            self.facial_expression_timing = 0  # 计时开始
            self.facial_expression_start_time = 0  # 开始时间
            self.facial_expression_limit_time = 2  # if >= 2 seconds, he/she is smiling
            self.count = 0
            self.sTime1 = 0
            self.interval1 = 0
            self.eTime1 = 0
            self.save_smile = '../../BJTU_2022Summer_Back_End/static/smile'
            # 初始化摄像头
            if not self.input_video:
                self.vs = cv2.VideoCapture(0)
                time.sleep(2)
            else:
                self.vs = cv2.VideoCapture(self.input_video)

            # 初始化人脸识别模型
            self.faceutil = FaceUtil(self.facial_recognition_model_path)
            self.facial_expression_model = load_model(self.facial_expression_model_path)
            # self.facial_expression_model = resnet_18()
            # self.facial_expression_model.load_weights(self.facial_expression_model)


            print('[INFO] 开始检测陌生人和表情...')
            # 不断循环
            self.counter = 0
        if(id==5):#facecollect
            self.audio_dir = '../audios'

            # 控制参数
            self.error = 0
            self.start_time = None
            self.limit_time = 2  # 2 秒

            self.action_list = ['blink', 'open_mouth', 'smile', 'rise_head', 'bow_head',
                           'look_left', 'look_right']
            self.action_map = {'blink': '请眨眼', 'open_mouth': '请张嘴',
                          'smile': '请笑一笑', 'rise_head': '请抬头',
                          'bow_head': '请低头', 'look_left': '请看左边',
                          'look_right': '请看右边'}
            # 设置摄像头
            self.vs = cv2.VideoCapture(0)
            self.vs.set(3, 640)  # set video widht
            self.vs.set(4, 480)  # set video height

            self.faceutil = FaceUtil()
            _, self.image = self.vs.read()
            self.action_name = ''
            self.counter=0

    def volunteerActivityDetection(self):
        self.counter += 1
        camera_turned = 0
        # grab the current frame
        (grabbed, frame) =self.vs.read()

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if self.input_video and not grabbed:
            return

        if not self.input_video:
            frame = cv2.flip(frame, 1)

        frame = imutils.resize(frame,
                               width=self.VIDEO_WIDTH,
                               height=self.VIDEO_HEIGHT)  # 压缩，为了加快识别速度

        face_location_list, names = self.faceutil.get_face_location_and_name(frame)

        # 得到画面的四分之一位置和四分之三位置，并垂直划线
        one_sixth_image_center = (int(self.VIDEO_WIDTH / 6), int(self.VIDEO_HEIGHT / 6))
        five_sixth_image_center = (int(self.VIDEO_WIDTH / 6 * 5),
                                   int(self.VIDEO_HEIGHT / 6 * 5))

        cv2.line(frame, (one_sixth_image_center[0], 0),
                 (one_sixth_image_center[0], self.VIDEO_HEIGHT),
                 (0, 255, 255), 1)
        cv2.line(frame, (five_sixth_image_center[0], 0),
                 (five_sixth_image_center[0], self.VIDEO_HEIGHT),
                 (0, 255, 255), 1)

        people_type_list = list(set([self.id_card_to_type[i] for i in names]))

        volunteer_name_direction_dict = {}
        volunteer_centroids = []
        old_people_centroids = []
        old_people_name = []

        # loop over the face bounding boxes
        for ((left, top, right, bottom), name) in zip(face_location_list, names):  # 处理单个人

            person_type = self.id_card_to_type[name]
            # 将人脸框出来
            rectangle_color = (0, 0, 255)
            if person_type == 'old_people':
                rectangle_color = (0, 0, 128)
            elif person_type == 'employee':
                rectangle_color = (255, 0, 0)
            elif person_type == 'volunteer':
                rectangle_color = (0, 255, 0)
            else:
                pass
            cv2.rectangle(frame, (left, top), (right, bottom),
                          rectangle_color, 2)

            if 'volunteer' not in people_type_list:  # 如果没有义工，直接跳出本次循环
                continue

            if person_type == 'volunteer':  # 如果检测到有义工存在
                # 获得义工位置
                volunteer_face_center = (int((right + left) / 2),
                                         int((top + bottom) / 2))
                volunteer_centroids.append(volunteer_face_center)

                cv2.circle(frame,
                           (volunteer_face_center[0], volunteer_face_center[1]),
                           8, (255, 0, 0), -1)

                adjust_direction = ''
                # face locates too left, servo need to turn right,
                # so that face turn right as well
                if volunteer_face_center[0] < one_sixth_image_center[0]:
                    adjust_direction = 'right'
                elif volunteer_face_center[0] > five_sixth_image_center[0]:
                    adjust_direction = 'left'

                volunteer_name_direction_dict[name] = adjust_direction

            elif person_type == 'old_people':  # 如果没有发现义工
                old_people_face_center = (int((right + left) / 2),
                                          int((top + bottom) / 2))
                old_people_centroids.append(old_people_face_center)
                old_people_name.append(name)

                cv2.circle(frame,
                           (old_people_face_center[0], old_people_face_center[1]),
                           4, (0, 255, 0), -1)
            else:
                pass

            # 人脸识别和表情识别都结束后，把表情和人名写上 (同时处理中文显示问题)
            img_PIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_PIL)
            final_label = self.id_card_to_name[name]
            draw.text((left, top - 30), final_label,
                      font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf', 40),
                      fill=(255, 0, 0))  # linux
            # 转换回OpenCV格式
            frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        # 义工追踪逻辑
        if 'volunteer' in people_type_list:
            volunteer_adjust_direction_list = list(volunteer_name_direction_dict.values())
            if '' in volunteer_adjust_direction_list:  # 有的义工恰好在范围内，所以不需要调整舵机
                print('%d-有义工恰好在可见范围内，摄像头不需要转动' % (self.counter))
            else:
                adjust_direction = volunteer_adjust_direction_list[0]
                camera_turned = 1
                print('%d-摄像头需要 turn %s %d 度' % (self.counter, adjust_direction, self.ANGLE))

        # 在义工和老人之间划线
        if camera_turned == 0:
            for i in volunteer_centroids:
                for j_index, j in enumerate(old_people_centroids):
                    pixel_distance = dist.euclidean(i, j)
                    face_pixel_width = sum([i[2] - i[0] for i in face_location_list]) / len(face_location_list)
                    pixel_per_metric = face_pixel_width / self.FACE_ACTUAL_WIDTH
                    actual_distance = pixel_distance / pixel_per_metric

                    if actual_distance < self.ACTUAL_DISTANCE_LIMIT:
                        cv2.line(frame, (int(i[0]), int(i[1])),
                                 (int(j[0]), int(j[1])), (255, 0, 255), 2)
                        label = 'distance: %dcm' % (actual_distance)
                        cv2.putText(frame, label, (frame.shape[1] - 150, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 255), 2)

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        event_desc = '%s正在与义工交互' % (self.id_card_to_name[old_people_name[j_index]])
                        event_location = '房间桌子'
                        print(
                            '[EVENT] %s, 房间桌子, %s 正在与义工交互.' % (
                            current_time, self.id_card_to_name[old_people_name[j_index]]))
                        cv2.imwrite(
                            os.path.join(self.output_activity_path,
                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                            frame)  # snapshot
                        cv2.imwrite(
                            os.path.join(self.save_activity,
                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                            frame)  # snapshot
                        str1 = "snapshot_" + time.strftime('%Y%m%d_%H%M%S') + ".jpg"
                        str = "http://127.0.0.1:5000/static/activity/" + str1
                        if (self.count == 0):
                            f = open('allowinsertdatabase.txt', 'w')
                            f.write('is_allowed=1')
                            f.close()
                        self.count += 1
                        # insert into database
                        command = '%s inserting.py --event_desc %s --event_type 1 --event_location %s --old_people_id %d --image %s' % (
                            self.python_path, event_desc, event_location, int(name),str)
                        p = subprocess.Popen(command, shell=True)
        else:
            self.count = 0
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def fenceDetection(self):

        ret, frame = self.vs.read()

        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if self.input_video and not ret:
            return

        if not self.input_video:
            frame = cv2.flip(frame, 1)

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=500)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # if the frame dimensions are empty, set them
        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []

        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if self.totalFrames % self.skip_frames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            self.trackers = []

            # convert the frame to a blob and pass the blob through the
            # network and obtain the detections
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (self.W, self.H), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()

            # loop over the detections
            for i in np.arange(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated
                # with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections by requiring a minimum
                # confidence
                if confidence > self.minimum_confidence:
                    # extract the index of the class label from the
                    # detections list
                    idx = int(detections[0, 0, i, 1])

                    # if the class label is not a person, ignore it
                    if self.CLASSES[idx] != "person":
                        continue

                    # compute the (x, y)-coordinates of the bounding box
                    # for the object
                    box = detections[0, 0, i, 3:7] * np.array([self.W, self.H, self.W, self.H])
                    (startX, startY, endX, endY) = box.astype("int")

                    # construct a dlib rectangle object from the bounding
                    # box coordinates and then start the dlib correlation
                    # tracker
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(startX, startY, endX, endY)
                    tracker.start_track(rgb, rect)

                    # add the tracker to our list of trackers so we can
                    # utilize it during skip frames
                    self.trackers.append(tracker)

        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in self.trackers:
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
        cv2.line(frame, (0, self.H // 2), (self.W, self.H // 2), (0, 255, 255), 2)

        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = self.ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = self.trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)
                self.count = 0
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
                    if direction < 0 and centroid[1] < self.H // 2:
                        self.totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > self.H // 2:
                        self.totalDown += 1
                        to.counted = True

                        self.current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        event_desc = '有人闯入禁止区域!!!'
                        event_location = '院子'
                        print('[EVENT] %s, 院子, 有人闯入禁止区域!!!' % (self.current_time))
                        cv2.imwrite(
                            os.path.join(self.output_fence_path, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                            frame)  # snapshot
                        cv2.imwrite(
                            os.path.join(self.save_fence, 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                            frame)  # snapshot
                        str1 = "snapshot_" + time.strftime('%Y%m%d_%H%M%S') + ".jpg"
                        str = "http://127.0.0.1:5000/static/fence/" + str1
                        if (self.count == 0):
                            f = open('allowinsertdatabase.txt', 'w')
                            f.write('is_allowed=1')
                            f.close()
                        self.count += 1
                        # insert into database
                        command = '%s inserting.py --event_desc %s --event_type 4 --event_location %s --image %s' % (
                            self.python_path, event_desc, event_location, str)
                        p = subprocess.Popen(command, shell=True)

                    # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to

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
            ("Down", self.totalDown),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, self.H - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # show the output frame
        # cv2.imshow("Prohibited Area", frame)
        #
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        self.totalFrames += 1
        self.fps.update()
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def facialExpressionAndStrangerDetection(self):

        self.counter += 1
        # grab the current frame
        (grabbed, frame) = self.vs.read()

        # if we are viewing a video and we did not grab a frame, then we
        # have reached the end of the video
        if self.input_video and not grabbed:
            return

        if not self.input_video:
            frame = cv2.flip(frame, 1)

        frame = imutils.resize(frame, width=self.VIDEO_WIDTH,
                               height=self.VIDEO_HEIGHT)  # 压缩，加快识别速度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale，表情识别

        face_location_list, names = self.faceutil.get_face_location_and_name(
            frame)

        # 得到画面的四分之一位置和四分之三位置，并垂直划线
        one_fourth_image_center = (int(self.VIDEO_WIDTH / 4),
                                   int(self.VIDEO_HEIGHT / 4))
        three_fourth_image_center = (int(self.VIDEO_WIDTH / 4 * 3),
                                     int(self.VIDEO_HEIGHT / 4 * 3))

        cv2.line(frame, (one_fourth_image_center[0], 0),
                 (one_fourth_image_center[0], self.VIDEO_HEIGHT),
                 (0, 255, 255), 1)
        cv2.line(frame, (three_fourth_image_center[0], 0),
                 (three_fourth_image_center[0], self.VIDEO_HEIGHT),
                 (0, 255, 255), 1)

        # 处理每一张识别到的人脸
        for ((left, top, right, bottom), name) in zip(face_location_list,
                                                      names):

            # 将人脸框出来
            rectangle_color = (0, 0, 255)
            if self.id_card_to_type[name] == 'old_people':
                rectangle_color = (0, 0, 128)
            elif self.id_card_to_type[name] == 'employee':
                rectangle_color = (255, 0, 0)
            elif self.id_card_to_type[name] == 'volunteer':
                rectangle_color = (0, 255, 0)
            else:
                pass
            cv2.rectangle(frame, (left, top), (right, bottom),
                          rectangle_color, 2)

            # 陌生人检测逻辑
            if 'Unknown' in names:  # alert
                self.sTime = 0
                if self.strangers_timing == 0:  # just start timing
                    self.strangers_timing = 1
                    self.strangers_start_time = time.time()
                else:  # already started timing
                    strangers_end_time = time.time()
                    difference = strangers_end_time - self.strangers_start_time

                    current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                 time.localtime(time.time()))

                    if difference < self.strangers_limit_time:
                        print('[INFO] %s, 房间, 陌生人仅出现 %.1f 秒. 忽略.' % (current_time, difference))
                    else:  # strangers appear
                        event_desc = '陌生人出现!!!'
                        event_location = '房间'
                        print('[EVENT] %s, 房间, 陌生人出现!!!' % (current_time))
                        cv2.imwrite(os.path.join(self.output_stranger_path,
                                                 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                    frame)  # snapshot
                        cv2.imwrite(os.path.join(self.save_stranger,
                                                 'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                    frame)  # snapshot
                        str1 = "snapshot_" + time.strftime('%Y%m%d_%H%M%S') + ".jpg"
                        str = "http://127.0.0.1:5000/static/stranger/" + str1
                        if (self.count1 == 0):
                            f = open('allowinsertdatabase.txt', 'w')
                            f.write('is_allowed=1')
                            f.close()
                        self.count1 += 1
                        # insert into database
                        command = '%s inserting.py --event_desc %s --event_type 2 --event_location %s --image %s' % (
                        self.python_path, event_desc, event_location,str)
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
                            print('%d-摄像头需要 turn %s %d 度' % (self.counter,
                                                             direction, self.ANGLE))

            else:  # everything is ok
                self.strangers_timing = 0
                if(self.sTime == 0):
                    self.sTime = time.time()
                    self.eTime = 0
                    self.interval = 0
                else:
                    self.eTime = time.time()
                    self.interval = self.eTime - self.sTime
                    if(self.interval > 5):
                        self.count1 = 0
                    else:
                        self.count1 = 0

            # 表情检测逻辑
            # 如果不是陌生人，且对象是老人
            if name != 'Unknown' and self.id_card_to_type[name] == 'old_people':
                # 表情检测逻辑
                roi = gray[top:bottom, left:right]
                roi = cv2.resize(roi, (self.FACIAL_EXPRESSION_TARGET_WIDTH,
                                       self.FACIAL_EXPRESSION_TARGET_HEIGHT))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # # determine facial expression
                # (neural, smile) = self.facial_expression_model.predict(roi)[0]
                # facial_expression_label = 'Neural' if neural > smile else 'Smile'

                labels = ['angry', 'disgust', 'fear', 'happy', 'normal', 'sad', 'surprise']
                print(self.facial_expression_model.predict(roi)[0])
                facial_expression_label = labels[self.facial_expression_model.predict(roi)[0].tolist().index(
                    max(self.facial_expression_model.predict(roi)[0].tolist()))]
                if facial_expression_label == 'happy':  # alert
                    self.sTime1 = 0
                    if self.facial_expression_timing == 0:  # just start timing
                        self.facial_expression_timing = 1
                        self.facial_expression_start_time = time.time()
                    else:  # already started timing
                        facial_expression_end_time = time.time()
                        difference = facial_expression_end_time - self.facial_expression_start_time

                        current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                                     time.localtime(time.time()))
                        if difference < self.facial_expression_limit_time:
                            print(
                                '[INFO] %s, 房间, %s仅笑了 %.1f 秒. 忽略.' % (current_time, self.id_card_to_name[name], difference))
                        else:  # he/she is really smiling
                            event_desc = '%s正在笑' % (self.id_card_to_name[name])
                            event_location = '房间'
                            print('[EVENT] %s, 房间, %s正在笑.' % (current_time, self.id_card_to_name[name]))
                            cv2.imwrite(os.path.join(self.output_smile_path,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                        frame)  # snapshot
                            cv2.imwrite(os.path.join(self.save_smile,
                                                     'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))),
                                        frame)  # snapshot
                            str1 = "snapshot_" + time.strftime('%Y%m%d_%H%M%S') + ".jpg"
                            str = "http://127.0.0.1:5000/static/smile/" + str1
                            if (self.count == 0):
                                f = open('allowinsertdatabase.txt', 'w')
                                f.write('is_allowed=1')
                                f.close()
                            self.count += 1
                            # insert into database
                            command = '%s inserting.py --event_desc %s --event_type 0 --event_location %s --old_people_id %d --image %s' % (
                            self.python_path, event_desc, event_location, int(name), str)
                            p = subprocess.Popen(command, shell=True)

                else:  # everything is ok
                    self.facial_expression_timing = 0
                    if (self.sTime1 == 0):
                        self.sTime1 = time.time()
                        self.eTime1 = 0
                        self.interval1 = 0
                    else:
                        self.eTime1 = time.time()
                        self.interval1 = self.eTime1 - self.sTime1
                        if (self.interval1 > 5):
                            self.count = 0
                        else:
                            self.count = 0

            else:  # 如果是陌生人，则不检测表情
                facial_expression_label = ''

            # 人脸识别和表情识别都结束后，把表情和人名写上
            # (同时处理中文显示问题)
            img_PIL = Image.fromarray(cv2.cvtColor(frame,
                                                   cv2.COLOR_BGR2RGB))

            draw = ImageDraw.Draw(img_PIL)
            final_label = self.id_card_to_name[name] + ': ' + self.facial_expression_id_to_name[
                facial_expression_label] if facial_expression_label else self.id_card_to_name[name]
            draw.text((left, top - 30), final_label,
                      font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf', 40),
                      fill=(255, 0, 0))  # linux

            # 转换回OpenCV格式
            frame = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
        # show our detected faces along with smiling/not smiling labels
        # cv2.imshow("Checking Strangers and Ole People's Face Expression",
        #            frame)
        #
        # # Press 'ESC' for exiting video
        # k = cv2.waitKey(1) & 0xff
        # if k == 27:
        #     break

    def faceCollect(self,imagedir,id,people):

        self.counter += 1
        _, self.image = self.vs.read()
        # if self.counter <= 10:  # 放弃前10帧
        #     return
        self.image = cv2.flip(self.image, 1)

        if self.error == 1:
            end_time = time.time()
            difference = end_time - self.start_time
            print(difference)
            if difference >= self.limit_time:
                self.error = 0

        face_location_list = self.faceutil.get_face_location(self.image)
        for (left, top, right, bottom) in face_location_list:
            cv2.rectangle(self.image, (left, top), (right, bottom),
                          (0, 0, 255), 2)

        # cv2.imshow('Collecting Faces', image)  # show the image
        # # Press 'ESC' for exiting video
        # k = cv2.waitKey(100) & 0xff
        # if k == 27:
        #     break

        face_count = len(face_location_list)
        if self.error == 0 and face_count == 0:  # 没有检测到人脸
            print('[WARNING] 没有检测到人脸')
            audioplayer.play_audio(os.path.join(self.audio_dir,
                                                'no_face_detected.mp3'))
            self.error = 1
            self.start_time = time.time()
        elif self.error == 0 and face_count == 1:  # 可以开始采集图像了
            print('[INFO] 可以开始采集图像了')
            audioplayer.play_audio(os.path.join(self.audio_dir,
                                                'start_image_capturing.mp3'))
            return
        elif self.error == 0 and face_count > 1:  # 检测到多张人脸
            print('[WARNING] 检测到多张人脸')
            audioplayer.play_audio(os.path.join(self.audio_dir,
                                                'multi_faces_detected.mp3'))
            self.error = 1
            self.start_time = time.time()
        else:
            pass
        ret, jpeg = cv2.imencode('.jpg', self.image)
        return jpeg.tobytes()


    def collectDir(self,imagedir,id):
        # 新建目录
        self.imagedir=imagedir
        self.id=str(id)
        if os.path.exists(os.path.join(imagedir, id)):
            print(imagedir + "路径已经存在")
        else:
            os.mkdir(os.path.join(imagedir, id))

    def coll(self,index):
        audioplayer.play_audio(os.path.join(self.audio_dir, self.action_list[index] + '.mp3'))
        self.action_name = self.action_map[self.action_list[index]]

        self.counter = 1

    def dealCollect(self,index,i):

        print('%s-%d' % (self.action_name, i))
        _, img_OpenCV = self.vs.read()
        img_OpenCV = cv2.flip(img_OpenCV, 1)
        origin_img = img_OpenCV.copy()  # 保存时使用

        face_location_list = self.faceutil.get_face_location(img_OpenCV)
        for (left, top, right, bottom) in face_location_list:
            cv2.rectangle(img_OpenCV, (left, top),
                          (right, bottom), (0, 0, 255), 2)

        img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV,
                                               cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(img_PIL)
        draw.text((int(self.image.shape[1] / 2), 30), self.action_name,
                  font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Unicode.ttf', 40),
                  fill=(255, 0, 0))  # linux

        # 转换回OpenCV格式
        img_OpenCV = cv2.cvtColor(np.asarray(img_PIL),
                                  cv2.COLOR_RGB2BGR)

        # cv2.imshow('Collecting Faces', img_OpenCV)  # show the image

        image_name = os.path.join(self.imagedir,self.id,
                                  self.action_list[index] + '_' + str(self.counter) + '.jpg')
        cv2.imwrite(image_name, origin_img)
        # Press 'ESC' for exiting video
        # k = cv2.waitKey(100) & 0xff
        # if k == 27:
        #     break
        self.counter += 1
        ret, jpeg = cv2.imencode('.jpg', img_OpenCV)
        return jpeg.tobytes()

    def getfall(self):
        arr=[]
        arr.append(self.mp_pose)
        arr.append(self.cap)
        return arr

    def fallDetection(self,pose):

    # while cap.isOpened():
        success, image = self.cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            return
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            for index, landmarks in enumerate(results.pose_landmarks.landmark):
                self.res_point.append(landmarks.x)
                self.res_point.append(landmarks.y)
                self.res_point.append(landmarks.z)
            shape1 = int(len(self.res_point) / len(self.keyXYZ))
            self.res_point = np.array(self.res_point).reshape(shape1, len(self.keyXYZ))
            pred = self.pose_knn.predict(self.res_point)
            self.res_point = []
            if pred == 0:
                event_desc = '有人摔倒!!!'
                event_location = '走廊'
                current_time = time.strftime('%Y-%m-%d %H:%M:%S',
                                             time.localtime(time.time()))
                print('[EVENT] %s, 走廊, 有人摔倒!!!' % (current_time))
                cv2.imwrite(os.path.join(self.output_fall_path,
                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), image)  # snapshot
                cv2.imwrite(os.path.join(self.save_fall,
                                         'snapshot_%s.jpg' % (time.strftime('%Y%m%d_%H%M%S'))), image)  # snapshot
                str1 = "snapshot_" + time.strftime('%Y%m%d_%H%M%S') + ".jpg"
                str = "http://127.0.0.1:5000/static/fall/" + str1
                if (self.count == 0):
                    f = open('allowinsertdatabase.txt', 'w')
                    f.write('is_allowed=1')
                    f.close()
                self.count += 1
                # insert into database
                command = '%s inserting.py --event_desc %s --event_type 3 --event_location %s --image %s' % (
                    self.python_path, event_desc, event_location,str)
                p = subprocess.Popen(command, shell=True)
                cv2.putText(image, "Fall", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
            else:
                cv2.putText(image, "Normal", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)
        else:
            self.count = 0
        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
        # Flip the image horizontally for a selfie-view display.
        currTime = time.time()
        fps = 1 / (currTime - self.prevTime)
        prevTime = currTime
        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
        # cv2.imshow('MediaPipe Pose', image)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    def __del__(self):
        print("release")
        self.vs.release()
        cv2.destroyAllWindows()