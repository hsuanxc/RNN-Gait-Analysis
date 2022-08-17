import argparse
import logging

import numpy as np
import sys

import tensorflow as tf
import random
from random import randint
from threading import Thread
import threading
import os
import cv2
import time

import json
from tf_pose.WebcamVideoStream import WebcamVideoStream
from tf_pose.networks import get_graph_path, model_wh
from tf_pose.tensblur.smoother import Smoother
from tf_pose.estimator import TfPoseEstimator
from tf_pose.common import CocoPart
from tf_pose import common


logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

#cam = WebcamVideoStream("rtsp://admin:dh123456@10.15.202.233:554").start()
#cam = WebcamVideoStream("rtsp://admin:abc12345@10.15.202.234:554").start() # hikvision
cam = WebcamVideoStream(0).start()
#cam = WebcamVideoStream("http://61.64.52.241/live?port=4935&app=demo&stream=1234").start()

#----- Set Parameters -----#
n_steps = 110    # 每40個frame為一次動作 (40)
n_input = 36    # 人體18個keypoint(x,y) (36)
n_hidden = 30   # 特徵的隱藏層數 (40)
n_classes = 12   # (3)

tf.Variable(0, trainable=False)


#----- Utility functions for training -----#
def LSTM_RNN(_X, _weights, _biases):
    
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    _X = tf.reshape(_X, [-1, n_input])   
    # Rectifies Linear Unit activation function used
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0) 

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    lstm_last_output = outputs[-1]
    
    # 線性激活
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


#----- Build the network -----#
# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input]) #None
y = tf.placeholder(tf.float32, [None, n_classes]) #None

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
saver = tf.train.Saver()

LABELS = [    #put your label here
    "Person_1",
    "Person_2",
    "Person_3",
    "Person_4",
    "Person_5",
    "Person_6",
    "Person_7",
	  "Person_8",
	  "Person_9",
	  "Person_10",
	  "Person_11",
	  "Person_12"
]
"""
LABELS = [    
    "Nobody",
    "Lie",
    "Sitting_on_the_bed",
    "Edge_of_bed"
]
"""
base_path = os.path.abspath(os.getcwd())

MODEL_PATH = base_path + "/data/walk_poses/lstm_model.ckpt"

class Terrain(object):
    def __init__(self):
        self.record_flag = False
        self.ods_flag = True
        self.i_num = 0
        self.results = "Nobody" #Unmanned
        self.statusTemporary = "Nobody"

    def run_lstm(self, X_val):
        s_time = time.time()
        X_ = np.array(X_val, dtype=np.float32)
        blocks = int(len(X_) / n_steps)
        X_ = np.array(np.split(X_, blocks))
        
        preds = sess.run([pred], feed_dict={ x: X_})
        prediction = preds[0].argmax(1)

        if preds[0][0][prediction] > 0.0:
            self.results = str(LABELS[prediction[0]])
            self.statusTemporary = str(LABELS[prediction[0]])
        else:
            self.results = self.statusTemporary

        e_time = time.time()
        #print("run_lstm time:%s (ms)"%str(round(((e_time-s_time)*1000), 2)))
        #print("preds:", preds[0][0])

    def update(self):
        print("--> strat updata")
        saver.restore(sess, MODEL_PATH)
        print("--> Model restored.")

        number = 0
        num_frames = 0
        X_val = []

        while True:
            ret, frame = cam.read()
            keylist = []
            imgcopy = frame.copy()
            time_start = time.time()
            #time.sleep(0.05)
            if ret == True:
                try:
                    humans = e.inference(frame, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                    keyImg, centers = TfPoseEstimator.draw_humans(frame, humans, imgcopy=False)
                    num_frames += 1
                except AssertionError:
                    print('body not in frame')
                
                # Store 18 key points in the list
                for i in range(18):
                    if i not in centers.keys():
                        keylist.append(0)
                        keylist.append(0)
                        continue
                    keylist.append(centers[i][0])
                    keylist.append(centers[i][1])
                
                # Run lstm model
                if len(X_val) < n_steps: #+1
                    X_val.append(keylist)
                else: 
                    X_val.pop(0)
                    X_val.append(keylist)
                    
                    number = number+1
                    if number%2 == 0:
                        thread_lstm = threading.Thread(target = self.run_lstm(X_val))
                        thread_lstm.start()

                    # 各參數依次是：照片/添加的文字/左上角坐標/字體/字體大小/顏色/字體粗細
                    cv2.putText(keyImg, self.results, (25,470), cv2.FONT_HERSHEY_COMPLEX, 1.6, (0,0,255), 4)
                time_end = time.time()
                fps = "FPS: " + str(round(1.0/(time_end-time_start), 2))                
                cv2.putText(keyImg, fps, (30,30), cv2.FONT_HERSHEY_COMPLEX, 0.8,(0,0,255),2)
                if self.record_flag == True:
                    video_out.write(keyImg)

                # Force all windows to close
                k = cv2.waitKey(1) & 0xFF
                if k == ord('p'):
                    cv2.imwrite('1_0%s.png'%self.i_num, imgcopy)
                    cv2.imwrite('0%s.png'%self.i_num, keyImg)
                    self.i_num = self.i_num + 1
                    print("save image")

                elif k == ord('s'):
                    if self.ods_flag == True:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_out = cv2.VideoWriter('%s.mp4'%time.strftime("%Y%m%d_%H%M%S", time.localtime()), fourcc, 10, (keyImg.shape[1], keyImg.shape[0]))
                        self.record_flag = True
                        self.ods_flag = False
                        print("Recording...")
                    else:
                        video_out.release()
                        #video_out = cv2.VideoWriter('%s.mp4'%time.strftime("%Y%m%d_%H%M%S", time.localtime()), fourcc, 12, (keyImg.shape[1], keyImg.shape[0]))
                        self.record_flag = False
                        self.ods_flag = True
                        print("Successfully save video")

                elif k == ord('q'):
                    print("--> break webkey")
                    break

            # show Recognition result
            cv2.imshow('keyImg', keyImg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')

    parser.add_argument('--camera', type=str, default='0') #rtsp://admin:123456@120.117.41.92:7070/stream1

    parser.add_argument('--resize', type=str, default='432x368',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')

    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='trt_tfopenpose_models_fp32', help='cmu / mobilenet_frozen-opt-model / mobilenet_v2_final, mobilenet_thin, mobilenet_v2_large, mobilenet_v2_small')

    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')

    parser.add_argument('--output_json', type=str, help='writing output json dir')

    parser.add_argument('--trt_bool', type=bool, help='Use TRT or Not', default=True)
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    w, h = model_wh(args.resize)
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=args.trt_bool)
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=args.trt_bool)
    logger.debug('cam read+')

    os.chdir('..')   
    t = Terrain()
    t.update()

    cam.stop()
    cv2.destroyAllWindows()
