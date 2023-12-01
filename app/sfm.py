import numpy as np
from data import dataset
import yaml
from tqdm import tqdm
from utils import *


class SFM():
    def __init__(self):
        self.file_path = r"../params.yaml"
        self.get_params()
        self.data = dataset(self.data_path)
        self.num_frame = self.data.num_frame
        # init matrices
        self.init_matrices()

    def get_params(self, ):
        try:
            with open(self.file_path, 'r') as yaml_file:
                self.params = yaml.safe_load(yaml_file)
                self._set_attributes()
            print("Parameters loaded successfully.")
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
        except Exception as e:
            print(f"Error loading parameters: {e}")

    def _set_attributes(self):
        if self.params:
            for key, value in self.params.items():
                setattr(self, key, value)

    def init_matrices(self):
        self.K = self.data.K
        self.R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.R_t_1 = np.empty((3, 4))
        self.P1 = np.matmul(self.K, self.R_t_0)
        self.P2 = np.empty((3, 4))
        self.pts_4d = []
        self.X = np.array([])
        self.Y = np.array([])
        self.Z = np.array([])

    def run(self):
        for idx in tqdm(range(self.num_frame)):
            if idx == 0:
                prev_img = self.data.next_imgs()
            else:
                curr_img = self.data.next_imgs()
                # Extract features
                prev_kp, prev_des = extract_features(
                    prev_img, detector=self.detector, GoodP=self.GoodP, )
                curr_kp, curr_des = extract_features(
                    curr_img, detector=self.detector, GoodP=self.GoodP, )
                # extract matches
                matches_unfilter = match_features(
                    des1=prev_des, des2=curr_des, matching=self.matching, detector=self.detector, )
                if self.dist_threshold is not None:
                    matches = filter_matches_distance(
                        matches_unfilter, dist_threshold=self.dist_threshold)
                else:
                    matches = matches_unfilter
                rmat, tvec, image1_points, image2_points = estimate_motion(
                    matches=matches, kp1=prev_kp, kp2=curr_kp, k=self.data.K, depth1=None)

                self.R_t_1[:3, :3] = np.matmul(rmat, self.R_t_0[:3, :3])
                self.R_t_1[:3, 3] = self.R_t_0[:3, 3] + np.matmul(self.R_t_0[:3, :3], tvec.ravel())

                self.P2 = np.matmul(self.K, self.R_t_1)


                prev_img = curr_img
                R_t_0 = np.copy(self.R_t_1)
                P1 = np.copy(self.P2)


if __name__ == "__main__":
    sfm = SFM()
    sfm.run()
