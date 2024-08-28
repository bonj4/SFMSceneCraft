import yaml
from tqdm import tqdm
from data import dataset
from display import Display3D
from utils import *


class SFM():
    def __init__(self):
        self.file_path = r"../params.yaml"
        self.get_params()
        self.data = dataset(self.data_path)
        self.num_frame = self.data.num_frame
        # init matrices
        self.init_matrices()
        self.disp = Display3D()

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
        self.P1 = np.eye(4)
        self.poses = []

        self.pts_4d = np.array([[], [], []]).T
        self.colors_4d = np.array([[], [], []]).T
        self.camera_parameters = np.zeros((self.num_frame, 6))
        self.camera_parameters[0, :] = proj_mat_to_camera_vec(self.P1)

    def run(self):

        for idx in tqdm(range(self.num_frame)):
            if idx == 0:
                prev_frame = Frame(self.data.next_imgs(), self.K)
                self.poses.append(prev_frame.pose)
            else:
                curr_frame = Frame(self.data.next_imgs(), self.K)
                # Extract features
                if not len(prev_frame.kps):
                    prev_frame.add_features(
                        extract_features(prev_frame.img, self.sift_peak_threshold,
                                         self.sift_edge_threshold, self.detector, self.GoodP, self.min_features))
                curr_frame.add_features(
                    extract_features(curr_frame.img, self.sift_peak_threshold,
                                     self.sift_edge_threshold, self.detector, self.GoodP, self.min_features))
                # extract matches
                matches = match_features(
                    prev_frame, curr_frame, matching=self.matching, detector=self.detector,
                    dist_threshold=self.dist_threshold)

                # rmat, tvec, image1_points, image2_points = estimate_motion(
                #     matches=matches, kp1=prev_frame.kp, kp2=curr_frame.kp, k=self.data.K, depth1=None)

                estimate_motion(prev_frame, curr_frame, matches)

                points_3d = triangulation(prev_frame, curr_frame)
                filtering_points(prev_frame, points_3d, self.max_dist)
                filtering_points(curr_frame, points_3d, self.max_dist)

                # TODO move poses,color_4d,pts_4d to display class. cause is those are use only in over there.
                self.poses.append(curr_frame.pose)
                self.pts_4d = np.concatenate((self.pts_4d, curr_frame.points3d))
                self.colors_4d = np.concatenate(
                    (self.colors_4d, curr_frame.get_colors(curr_frame.untriangulate_points2d)))
                self.disp.paint(self.poses, self.pts_4d, self.colors_4d)
                prev_frame = curr_frame


if __name__ == "__main__":
    sfm = SFM()
    sfm.run()
