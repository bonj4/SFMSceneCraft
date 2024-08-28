import numpy as np


class Frame():
    def __init__(self, imgs, K):
        # create figure
        self.img = imgs
        self.kps = []
        self.des = []
        self.untriangulate_points2d = np.array([[], []]).T
        self.triangulated_points2d = np.array([[], []]).T
        self.points3d = np.array([[], [], []]).T
        self.pose = np.eye(4)
        self.K = K
        self.W = self.img.shape[1]
        self.H = self.img.shape[0]

    def add_features(self, features):
        # if self.kp_left is None:
        self.kps = features[0]
        self.des = features[1]

    # normalized keypoints
    # didnt use anywhere
    def normalize_points(self, points2d):
        self.add_points2d(points2d)
        return np.dot(np.linalg.inv(self.K),
                      np.concatenate([self.points2d, np.ones((self.points2d.shape[0], 1))], axis=1).T).T[:, 0:2]

    def calc_pose(self, rmat, tvec,prev_pose=None):
        def poseRt(R, t):
            ret = np.eye(4)
            ret[:3, :3] = R
            ret[:3, 3] = t.T
            return ret

        Rt = poseRt(rmat, tvec)
        if prev_pose is not None :self.pose = np.dot(prev_pose, np.linalg.inv(Rt))
        else: self.pose = np.linalg.inv(Rt)


    def get_colors(self, kps):
        return np.array([self.img[int(kp[1]), int(kp[0])] for kp in kps])
