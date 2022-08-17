import argparse
import logging
import tensorflow as tf
import random
from random import randint
from threading import Thread
import os, sys
import cv2
import time
import json
from tf_pose.WebcamVideoStream import WebcamVideoStream
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.tensblur.smoother import Smoother
from tf_pose.estimator import TfPoseEstimator
from tf_pose.common import CocoPart
from tf_pose import common

import pickle
import xlrd
import xlwt
import datetime

row_num = 0
base_path = os.path.abspath(os.getcwd())
file_path = base_path + "/data/walk_poses/"
video_path = base_path + "/data/videos/"
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
cam = WebcamVideoStream(0).start()
n_steps = 110


class Terrain(object):
    def __init__(self):
        self.record_flag = False
        self.flag = True
        self.odsflag = False
        self.i_num = 0
        self.A_date = 0
        self.Sw_date = 0
        self.isTestset = False
        print("--> strat init")
        # setup datetime
        self.ISOTIMEFORMAT_YMD = '%Y%m%d'
        self.ISOTIMEFORMAT_HMS = '%H:%M:%S, %f'
        print("--> Successfully setup datetime")

        # setup xls flip
        self.xlwtBook = xlwt.Workbook()
        self.keySheet_x = self.xlwtBook.add_sheet('KeyHistory_x')
        self.keySheet_x.write(0,0,'Time')
        self.keySheet_x.col(0).width = 7500
		
        self.keySheet_y = self.xlwtBook.add_sheet('KeyHistory_y')
        self.keySheet_y.write(0,0,'Time')
        self.keySheet_y.col(0).width = 7500

        # open file
        self.keyText_X = open(file_path + "X_train.txt", "a+")
        self.keyText_Y = open(file_path + "Y_train.txt", "a+")
        self.keyText_X_test = open(file_path + "X_test.txt", "a+")
        self.keyText_Y_test = open(file_path + "Y_test.txt", "a+")
        for i in range(18):
            self.keySheet_x.write(0,i+1,'keypoints ' + str(i))
            self.keySheet_x.col(i+1).width = 5000
            self.keySheet_y.write(0,i+1,'keypoints ' + str(i))
            self.keySheet_y.col(i+1).width = 5000
        print("--> Successfully setup xls flip")


    def update(self):
        print("--> strat updata")
        num_frames = 0
        A=0
        while True:
            ret, frame = cam.read()
            time_start = time.time()
            #time.sleep(0.05)
            if ret == True:
                try:
                    humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                    keyImg, centers = TfPoseEstimator.draw_humans(frame, humans, imgcopy=True)
                except AssertionError:
                    print('body not in frame')
                # Force all windows to close
                k = cv2.waitKey(1) & 0xFF
                if k == ord('s'):
                    if self.flag == True:
                        self.keyText_X = open(file_path + "X_train.txt", "a+")
                        self.keyText_Y = open(file_path + "Y_train.txt", "a+")
                        self.odsflag = True                       
                        self.flag = False
                        self.Sw_date = self.Sw_date + 1

                        self.record_flag = True
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_out = cv2.VideoWriter(video_path + 'train_person{}_record{}.mp4'.format(str(self.A_date), str(self.Sw_date)), fourcc, 11, (frame.shape[1], frame.shape[0]))

                        print("--> recording training keypoint data...")
                elif k == ord('t'):
                    if self.flag == True:
                        self.isTestset = True
                        self.keyText_X_test = open(file_path + "X_test.txt", "a+")
                        self.keyText_Y_test = open(file_path + "Y_test.txt", "a+")
                        self.odsflag = True
                        self.flag = False
                        self.Sw_date = self.Sw_date + 1

                        self.record_flag = True
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_out = cv2.VideoWriter(video_path + 'test_person{}_record{}.mp4'.format(str(self.A_date), str(self.Sw_date)), fourcc, 11, (frame.shape[1], frame.shape[0]))

                        print("--> recording testing keypoint data...")
                elif k == ord('z'): # next person
                    if self.A_date == 10 :
                            self.A_date = 0
                    else:
                            self.A_date = self.A_date+1
                elif k == ord('x'): # last person
                    if self.A_date == 0 :
                            self.A_date = 10
                    else:
                            self.A_date = self.A_date-1
                elif k == ord('0'): # reset file count
                    self.Sw_date = 0
                elif k == ord('1'): # file count + 1
                    self.Sw_date = self.Sw_date + 1 
                elif k == ord('2'): # file count - 1
                    self.Sw_date = self.Sw_date - 1 
                elif k == ord('p'): # save a picture
                    cv2.imwrite('0%s.png'%self.i_num, keyImg)
                    self.i_num = self.i_num + 1
                elif k == ord('q'): # quit
                    print("--> break webkey")
                    break


                if self.odsflag == True:
                    keylist = []
                    for i in range(18):
                        if i not in centers.keys():
                            keylist.append(0)
                            keylist.append(0)
                            continue
                        keylist.append(centers[i][0])
                        keylist.append(centers[i][1])

                    A=A+1
                    # write a line at the end of the file
                    textData = str(keylist)[1:len(str(keylist))-1]
                    
                    if A is not n_steps+1:
                        if self.isTestset:
                            self.keyText_X_test.seek(0, 2)
                            self.keyText_X_test.write(textData.replace(" ", "") + "\n")
                            if self.record_flag == True:
                                video_out.write(frame)
                        else:
                            self.keyText_X.seek(0,2)
                            self.keyText_X.write(textData.replace(" ", "") + "\n")
                            if self.record_flag == True:
                                video_out.write(frame)
                    else:
                        if self.isTestset:
                            self.keyText_Y_test.seek(0, 2)
                            self.keyText_Y_test.write(str(self.A_date) + "\n")
                            self.record_flag = False
                            self.isTestset = False
                            video_out.release()
                        else:
                            self.keyText_Y.seek(0,2)
                            self.keyText_Y.write(str(self.A_date) + "\n")
                            self.record_flag = False
                            video_out.release()

                        self.flag = True                      
                        self.odsflag = False
                        self.keyText_X.close()
                        self.keyText_X_test.close()
                        self.keyText_Y_test.close()
                        self.keyText_Y.close()
                        A=0
                        print("--> Successfully save keypoint data")

            # show Recognition result
            time_end = time.time()
            fps = "FPS: " + str(round(1.0/(time_end-time_start), 2))
            A_DATE = "    Digital: " + str(self.A_date) + "   SW :" + str(self.Sw_date)
            Txt_All = fps + A_DATE
            cv2.putText(keyImg, Txt_All, (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.8,(0,0,255),2)
            #cv2.namedWindow("keyImg",cv2.WINDOW_NORMAL)
            #tt = cv2.resize(keyImg,(100,100))
            cv2.imshow('keyImg', keyImg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')

    parser.add_argument('--camera', type=str, default='0')

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')

    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='trt_tfopenpose_models_fp32', help='trt_tfopenpose_models_fp16') # trt_tfopenpose_models_fp32

    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--output_json', type=str, help='writing output json dir')  #trt_bool

    parser.add_argument('--trt_bool', type=bool, help='Use TRT or Not', default=True)

    args = parser.parse_args()
    print(args)
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    
    w, h = model_wh(args.resize)
    
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w ,h), trt_bool=args.trt_bool)
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=args.trt_bool)
    logger.debug('cam read+')

    os.chdir('..')
    print('width',w,'height',h)
    t = Terrain()
    t.update()

    cam.stop()
    cv2.destroyAllWindows()
