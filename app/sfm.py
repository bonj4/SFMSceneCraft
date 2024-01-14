from display import Display3D
from data import dataset
import yaml
from tqdm import tqdm
from utils import *
from scipy.optimize import least_squares
from bundle_adjustment import bundle_adjustment_sparsity, fun


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
        self.R_t_0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.R_t_1 = np.empty((3, 4))
        self.P1 = np.eye(4)

        self.pts_4d = np.array([[], [], []]).T
        self.colors_4d = np.array([[], [], []]).T
        self.poses = [self.P1]
        self.point_indices = []
        self.camera_indices = []
        self.camera_parameters = np.zeros((self.num_frame, 6))
        self.camera_parameters[0, :] = proj_mat_to_camera_vec(self.P1)

    def run(self):

        for idx in tqdm(range(self.num_frame)):
            if idx == 0:
                prev_img = self.data.next_imgs()
            else:
                curr_img = self.data.next_imgs()
                # Extract features
                prev_kp, prev_des = extract_features(
                    prev_img, detector=self.detector, GoodP=self.GoodP, sift_peak_threshold=self.sift_peak_threshold,
                    edgeThreshold=self.sift_edge_threshold,min_features=self.min_features)
                curr_kp, curr_des = extract_features(
                    curr_img, detector=self.detector, GoodP=self.GoodP, sift_peak_threshold=self.sift_peak_threshold,
                    edgeThreshold=self.sift_edge_threshold,min_features=self.min_features)
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

                Rt = poseRt(rmat, tvec)
                self.P2 = np.dot(self.P1, np.linalg.inv(Rt), )
                self.poses.append(self.P2)

                curr_clr = get_colors(curr_img, image2_points)
                point1, point2 = norm_points(image1_points, image2_points, self.data.K)

                points_3d = triangulatecv(self.P1, self.P2, point1, point2)

                points_3d = cv2.convertPointsFromHomogeneous(points_3d)
                dist_list = calc_dist(origin=self.P2[:3, 3], ls=points_3d[:, 0])
                filter_pts = points_3d[:, 0, 2] > 0
                filter_pts &= dist_list < self.max_dist
                filtered_points_3d = points_3d[filter_pts][:, 0]
                self.pts_4d = np.concatenate((self.pts_4d, filtered_points_3d))

                filtered_colors = curr_clr[filter_pts]
                self.colors_4d = np.concatenate((self.colors_4d, filtered_colors))

                print("observed point: ", filtered_points_3d.shape, " 3d points: ", self.pts_4d.shape)

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
                self.disp.paint(self.poses, self.pts_4d, self.colors_4d)
                prev_img = curr_img
                # self.P1 = np.copy(recover_projection_matrix(camera_params[-1]))
                self.P1 = np.copy(self.P2)


if __name__ == "__main__":
    sfm = SFM()
    sfm.run()
