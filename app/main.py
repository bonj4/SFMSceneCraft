import pandas as pd
import cv2 
import numpy as np
import os
from utils import *
from data import dataset
import time
from plot_utils import viz_3d
from bundle_adjustment import bundle_adjustment

def SFM(detector='sift', matching='FLANN',GoodP=True, dist_threshold=0.35,):
    data=dataset()
    num_frame = len(os.listdir(data.sequences_dir+"images"))
    K=data.K
    R_t_0 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    R_t_1 = np.empty((3,4))
    P1 = np.matmul(K, R_t_0)
    P2 = np.empty((3,4))
    pts_4d = []

    X = np.array([])
    Y = np.array([])
    Z = np.array([])
    
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
            
            
            R_t_1[:3,:3] = np.matmul(rmat, R_t_0[:3,:3])
            R_t_1[:3, 3] = R_t_0[:3, 3] + np.matmul(R_t_0[:3,:3],tvec.ravel())
            
            P2 = np.matmul(K, R_t_1)
            
            image1_points = np.transpose(image1_points)
            image2_points = np.transpose(image2_points)
            
            points_3d = cv2.triangulatePoints(P1, P2, image1_points, image2_points)
            points_3d /= points_3d[3]
            
            # P2, points_3d = bundle_adjustment(points_3d, image2_points, img, P2)
            
            X = np.concatenate((X, points_3d[0]))
            Y = np.concatenate((Y, points_3d[1]))
            Z = np.concatenate((Z, points_3d[2]))
            
            prev_img=img
            R_t_0 = np.copy(R_t_1)
            P1 = np.copy(P2)

    pts_4d.append(X)
    pts_4d.append(Y)
    pts_4d.append(Z)
    viz_3d(np.array(pts_4d))
if __name__=='__main__':
    s=time.perf_counter()
    SFM()
    cv2.destroyAllWindows()
    print(time.perf_counter()-s)