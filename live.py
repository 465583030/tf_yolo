#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
from time import time
from yolo import Yolo
from imager import *

imer = Imager()
yl= Yolo()
cam = cv2.VideoCapture(0)
frame_interval = 1
fps_display_interval = 5
frame_rate = 0
frame_count = 0
start_time = time()

while True:
    _, frame = cam.read()

    if frame_count % frame_interval == 0:
        imer.imset(frame)
        input_list = imer.preproces()
        pred_list = yl.predict(input_list)
        frame = imer.visualise_preds(pred_list)[0]

    duration = time() - start_time
    if duration > fps_display_interval:
        frame_rate = int(frame_count/duration)
        start_time = time()
        frame_count = 0
        print('fps: {}'.format(frame_rate))
        
    frame_count += 1
    cv2.imshow('yolo v3', frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


