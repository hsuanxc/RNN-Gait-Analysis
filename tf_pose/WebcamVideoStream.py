# import the necessary packages
from threading import Thread
import cv2, time
import math


class WebcamVideoStream:
    def __init__(self, src=0, resolution=(640, 480), FPS=30, name="WebcamVideoStream" ,cutted = 100):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.stream.set(cv2.CAP_PROP_FPS,FPS)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH,int(resolution[0]))
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT,int(resolution[1]))
        #print("Width:%s, Height:%s, FPS:%s"%(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH),
                #self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT),self.stream.get(cv2.CAP_PROP_FPS)))
        self.cutted = cutted
        self.crop_img = []
        (self.grabbed, self.frame) = self.stream.read()


        # initialize the thread name
        self.name = name

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, name=self.name, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
        # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

        # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        if self.grabbed:
            return self.grabbed, self.frame
        else:
            return self.grabbed, None


    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
