from display import Display3D
from data import dataset
import yaml
from tqdm import tqdm
from utils import *
from scipy.optimize import least_squares
from bundle_adjustment import bundle_adjustment_sparsity, fun
from frame import Frame


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

                estimate_motion(prev_frame, curr_frame, self.K, matches)

                points_3d = triangulation(prev_frame, curr_frame, self.K)
                filtering_points(prev_frame, points_3d, self.max_dist)
                filtering_points(curr_frame, points_3d, self.max_dist)


                # Bundle adjustment TODO make pure and understoodable
                # n_cameras, n_points, curr_points = len(self.poses), len(self.pts_4d), len(filtered_points_3d)
                # if idx == 1:
                #     self.points_2d = point1[filter_pts]
                #     self.camera_indices.extend([0] * n_points)
                #     self.point_indices.extend(np.arange(n_points))
                #
                # self.camera_indices.extend([idx] * curr_points)
                # self.point_indices.extend(np.arange(curr_points))
                # self.points_2d = np.vstack((self.points_2d, point2[filter_pts]))
                #
                # self.camera_parameters[idx, :] = proj_mat_to_camera_vec(self.P2)
                #
                # x0 = np.hstack(
                #     (self.camera_parameters[:idx + 1].ravel(), self.pts_4d.ravel()))
                # A = bundle_adjustment_sparsity(n_cameras, n_points, np.array(self.camera_indices),
                #                                np.array(self.point_indices))
                # res = least_squares(fun, x0, jac='3-point', jac_sparsity=A, verbose=1, x_scale='jac', ftol=1e-5,
                #                     method='trf', loss='soft_l1',
                #                     args=(
                #                         n_cameras, n_points, np.array(self.camera_indices),
                #                         np.array(self.point_indices),
                #                         self.points_2d,
                #                         self.data.K))
                # optimized_params = res.x
                # camera_params = optimized_params[:n_cameras * 6].reshape((n_cameras, 6))
                # self.pts_4d = optimized_params[n_cameras * 6:].reshape((n_points, 3))
                # for idx, camera_param in enumerate(camera_params):
                #     self.poses[idx] = recover_projection_matrix(camera_param)
                #     self.camera_parameters[idx, :] = camera_param
                self.poses.append(curr_frame.pose)
                self.pts_4d = np.concatenate((self.pts_4d, curr_frame.points3d))
                self.colors_4d = np.concatenate(
                    (self.colors_4d, curr_frame.get_colors(curr_frame.untriangulate_points2d)))
                self.disp.paint(self.poses, self.pts_4d, self.colors_4d)
                prev_frame = curr_frame


if __name__ == "__main__":
    sfm = SFM()
    sfm.run()
