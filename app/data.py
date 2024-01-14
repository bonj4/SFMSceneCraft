import pandas as pd
import cv2 
import numpy as np
import os

class dataset():
    def __init__(self,data_path:str):

        # deprecated
        # self.sequences_dir = 'data/{}/'.format(model_name)
        self.sequences_dir=data_path

        self.img=(cv2.imread(self.sequences_dir+"images/"+i,) for i in sorted(os.listdir(self.sequences_dir+"images")) if i.endswith("JPG"))
        self.num_frame = len(os.listdir(self.sequences_dir+"images"))
        self.K=self.get_camera_intrinsic_params(self.sequences_dir+'cameras.txt')
        
    def next_imgs(self):
        return next(self.img)
    def reset_images(self):
        self.img=(cv2.imread(self.sequences_dir+"images/"+i,0) for i in sorted(os.listdir(self.sequences_dir+"images")) if i.endswith("JPG"))
    # def get_camera_intrinsic_params(self,path):
    #     K = []
    #     with open(path) as f:
    #         lines = f.readlines()
    #         print(lines)
    #         calib_info = [float(val) for val in lines[0].split(' ')]
    #         row1 = [calib_info[0], calib_info[1], calib_info[2]]
    #         row2 = [calib_info[3], calib_info[4], calib_info[5]]
    #         row3 = [calib_info[6], calib_info[7], calib_info[8]]
    #
    #         K.append(row1)
    #         K.append(row2)
    #         K.append(row3)
    #     return K
    def get_camera_intrinsic_params(self,path):
        self.k_flatten=pd.read_csv(path, delimiter=' ', header=None, index_col=0)
        return np.array(self.k_flatten).reshape((3,3))

