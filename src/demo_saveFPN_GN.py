from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import time
import cv2
import os
import time
import numpy as np


from opts import opts
from detector_factory import detector_factory


image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']



def save_result(ret,file):
    P_list=[]
    B_list=[]
    C_list=[]
    
    result_p=ret[1]
    result_b=ret[2]
    result_c=np.concatenate((ret[3],ret[4],ret[6],ret[8]),axis=0)
    
    for i in range(len(result_p)):
        if result_p[i][4] >= 0.3:
            P_list.append(result_p[i])
    
    for i in range(len(result_b)):
        if result_b[i][4] >= 0.3:
            B_list.append(result_b[i])

    for i in range(len(result_c)):
        if result_c[i][4] >= 0.3:
            C_list.append(result_c[i])
    
    with open('rgb_detection_train_waymo.txt', 'a') as f:
        for item in P_list:
            f.write("dataset/KITTI/object/training/image_2/" + file + " %d %.6f %d %d %d %d\n" % (1,item[4],item[0],item[1],item[2],item[3])) 
        for item in B_list:
            f.write("dataset/KITTI/object/training/image_2/" + file + " %d %.6f %d %d %d %d\n" % (3,item[4],item[0],item[1],item[2],item[3])) 
        for item in C_list:
            f.write("dataset/KITTI/object/training/image_2/" + file + " %d %.6f %d %d %d %d\n" % (2,item[4],item[0],item[1],item[2],item[3])) 

    


def demo(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 0)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
    detector.pause = False
    while True:
        _, img = cam.read()
        cv2.imshow('input', img)
        ret = detector.run(img)
        time_str = ''
        for stat in time_stats:
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        print(time_str)
        if cv2.waitKey(1) == 27:
            return  # esc to quit
  else:
    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
      image_names = [opt.demo]
    
    i=0
    for (image_name) in image_names:
      file_name=image_name.split('/')[-1]
      ret = detector.run(image_name)['results']
      
      save_result(ret,file_name)
           
      print(i)
      i+=1

if __name__ == '__main__':
  opt = opts().init()
  demo(opt)
