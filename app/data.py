import pandas as pd
import cv2 
import numpy as np
import os

class dataset():
    def __init__(self,seq="00"):
        
        self.sequences_dir = 'sequences/{}/'.format(seq)
        self.poses_dir = 'poses/{}.txt'.format(seq)
        
        self.left_img=(cv2.imread(self.sequences_dir+"image_0/"+i,0) for i in sorted(os.listdir(self.sequences_dir+"image_0")) if i.endswith("png"))
        self.right_img=(cv2.imread(self.sequences_dir+"image_1/"+i,0) for i in sorted(os.listdir(self.sequences_dir+"image_1")) if i.endswith("png"))
        self.poses=pd.read_csv(self.poses_dir, delimiter=' ', header=None)
        self.calib=pd.read_csv(self.sequences_dir+"calib.txt", delimiter=' ', header=None, index_col=0)
        self.times=pd.read_csv(self.sequences_dir+"times.txt",).to_numpy()
        
        self.p0=np.array(self.calib.iloc[0]).reshape((3,4))
        self.p1=np.array(self.calib.iloc[1]).reshape((3,4))
        self.Tr = np.array(self.calib.iloc[-1]).reshape((3,4))

        self.gt = np.zeros((len(self.poses), 3, 4))
        for i in range(len(self.poses)):
            self.gt[i] = np.array(self.poses.iloc[i]).reshape((3, 4))
        
    def next_imgs(self):
        return next(self.left_img),next(self.right_img)
    def reset_images(self):
        self.left_img=(cv2.imread(self.sequences_dir+"image_0/"+i,0) for i in sorted(os.listdir(self.sequences_dir+"image_0")) if i.endswith("png"))
        self.right_img=(cv2.imread(self.sequences_dir+"image_1/"+i,0) for i in sorted(os.listdir(self.sequences_dir+"image_1")) if i.endswith("png"))