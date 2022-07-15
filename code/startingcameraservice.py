# -*- coding: utf-8 -*-

'''
启动摄像头主程序

用法: 
python startingcameraservice.py
python startingcameraservice.py --location room

直接执行即可启动摄像头，浏览器访问 http://192.168.1.156:5001/ 即可看到
摄像头实时画面

'''
import argparse
import datetime
import json
import time

import cv2
from flask import Flask, render_template, Response, request
from flask_cors import CORS
from CV_Service import cv


# 传入参数
# ap = argparse.ArgumentParser()
# ap.add_argument("-f", "--location", required=False,
#                 default = 'room', help="")
# args = vars(ap.parse_args())
# location = args['location']
#
# if location not in ['room', 'yard', 'corridor', 'desk']:
#     raise ValueError('location must be one of room, yard, corridor or desk')

# API
app = Flask(__name__)
CORS(app, resources=r'/*')
video_camera = None
global_frame = None
from FileService import updatePeopleInfo
import pymysql #导入模块


connect = pymysql.connect(host='localhost',  # 本地数据库
                                   user='root',
                                   password='jiangshan201310.',
                                   db='old_care',
                                   charset='utf8')  # 服务器名,账户,密码，数据库名称
cursor = connect.cursor()
@app.route('/')
def index():
    return Response(gen_facial(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/record_status', methods=['POST'])
# def record_status():
#     global video_camera
#     if video_camera == None:
#         video_camera = VideoCamera()
#
#     status = request.form.get('status')
#     save_video_path = request.form.get('save_video_path')
#
#     if status == "true":
#         video_camera.start_record(save_video_path)
#         return 'start record'
#     else:
#         video_camera.stop_record()
#         return 'stop record'
#
# class VideoCamera(object):
#     def __init__(self):
#         # Using OpenCV to capture from device 0. If you have trouble capturing
#         # from a webcam, comment the line below out and use a video file
#         # instead.
#         self.video = cv2.VideoCapture("http://admin:admin@192.168.1.14:8081/")
#         # If you decide to use video.mp4, you must have this file in the folder
#         # as the main.py.
#         # self.video = cv2.VideoCapture('video.mp4')
#
#     def __del__(self):
#         self.video.release()
#
#     def get_frame(self):
#         success, image = self.video.read()
#         image = cv2.flip(image, 1)
#         # We are using Motion JPEG, but OpenCV defaults to capture raw images,
#         # so we must encode it into JPEG in order to correctly display the
#         # video stream.
#         ret, jpeg = cv2.imencode('.jpg', image)
#         return jpeg.tobytes()
#
# def video_stream():
#
#
#     while True:
#         frame = VideoCamera().get_frame()
#
#         if frame is not None:
#             global_frame = frame
#             yield (b'--frame\r\n'
#                     b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#         else:
#             yield (b'--frame\r\n'
#                             b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
#
# @app.route('/video_viewer')
# def video_viewer():
#     return Response(video_stream(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
def gen(camera):
    global video_camera
    global global_frame
    start=datetime.datetime.now()
    while True:
        frame = camera.volunteerActivityDetection()

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
        end=datetime.datetime.now()
        if((end-start).hours > 24):
            del camera
            break

@app.route('/desk')#desk-老人-义工交互
def volunteerDetection():
    return Response(gen(cv(1)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_fence(camera):
    global video_camera
    global global_frame
    start = datetime.datetime.now()
    while True:
        frame = camera.fenceDetection()

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
        end = datetime.datetime.now()
        if ((end - start).hours > 24):
            del camera
            break
@app.route('/yard')#yard-禁入区域
def fenceDetect():
    return Response(gen_fence(cv(3)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_fall(camera):
    global video_camera
    global global_frame
    arr=camera.getfall()
    start = datetime.datetime.now()
    with arr[0].Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while arr[1].isOpened():
            frame = camera.fallDetection(pose)

            if frame is not None:
                global_frame = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
            end = datetime.datetime.now()
            if ((end - start).hours > 24):
                del camera
                break

@app.route('/corridor')#corridor-走廊-摔倒
def fallDetect():
    return Response(gen_fall(cv(2)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def gen_facial(camera):
    global video_camera
    global global_frame
    start = datetime.datetime.now()
    while True:
        frame = camera.facialExpressionAndStrangerDetection()

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
        end = datetime.datetime.now()
        if ((end - start).hours > 24): #每天重启一次系统
            del camera
            break

@app.route('/room')#room-陌生人出现&老人笑了
def facialExpressionAndStrangerDetect():
    return Response(gen_facial(cv(4)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
def gen_coll(camera,people,id):
    global video_camera
    global global_frame

    dir="../images/user/"+people
    print(dir)
    # camera.faceCollect(dir, str(id))
    while True:
        frame = camera.faceCollect(dir,str(id),people)

        if frame is not None:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break
    camera.collectDir(dir, str(id))
    for i in range(0,7):
        camera.coll(i)

        for j in range(15):
            frame=camera.dealCollect(i,j)
            if frame is not None:
                global_frame = frame
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
    name = ''
    if people == 'old_people':
        select = "select username from oldperson_info where id = '{}'".format(id)
        cursor.execute(select)
        result = cursor.fetchall()
        name = result[0][0]
    elif people == 'emplyees':
        select = "select username from employee_info where id = '{}'".format(id)
        cursor.execute(select)
        result = cursor.fetchall()
        name = result[0][0]
    else:
        select = "select name from employee_info where id = '{}'".format(id)
        cursor.execute(select)
        result = cursor.fetchall()
        name = result[0][0]
    print(name,id,people)
    updatePeopleInfo(id, name, people)
    del(camera)

@app.route('/faceCollectOld')#人脸采集
def faceCollOld():
    id = request.args.get("id")
    print(str(id))
    return Response(gen_coll(cv(5),'old_people',id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/faceCollectemp')#人脸采集
def faceCollemp():
    id = request.args.get("id")
    return Response(gen_coll(cv(5),'employees',id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/faceCollectvol')#人脸采集
def faceCollvol():
    id = request.args.get("id")
    return Response(gen_coll(cv(5),'volunteers',id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5001,threaded = False)
