import pandas as pd
import cv2 
import numpy as np
import os
from utils import *
from data import dataset
import time

def SFM(detector='orb', matching='BF',GoodP=True, dist_threshold=0.5,):
    data=dataset()
    num_frame = len(os.listdir(data.sequences_dir+"images"))
    K=data.K
    R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    R_t_1 = np.empty((3,4))
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3,4))
    for idx in range(num_frame):
        if idx==0:
            prev_img=data.next_imgs()
        else:
            img=data.next_imgs()
            # Extract features
            prev_kp, prev_des = extract_features(
                prev_img, detector=detector, GoodP=GoodP,)
            img_kp, img_des = extract_features(
                img, detector=detector, GoodP=GoodP,)
            # extract matches
            matches_unfilter = match_features(
                des1=prev_des, des2=img_des, matching=matching, detector=detector,)
            # filtering the matches
            if dist_threshold is not None:
                matches = filter_matches_distance(
                    matches_unfilter, dist_threshold=dist_threshold)
            else:
                matches = matches_unfilter
            rmat, tvec, image1_points, image2_points = estimate_motion(
                    matches=matches, kp1=prev_kp, kp2=img_kp, k=data.K, depth1=None)
            
            prev_img=img
            R_t_0 = np.copy(R_t_1)
            P1 = np.copy(P2)

    
if __name__=='__main__':
    s=time.perf_counter()
    SFM()
    cv2.destroyAllWindows()
    print(time.perf_counter()-s)