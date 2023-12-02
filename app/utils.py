import numpy as np
import cv2
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from random import randint
from mayavi import mlab


def mse(ground_truth, estimated):
    nframes_est = estimated.shape[0]

    se = [((es[0, 3] - gt[0, 3]) ** 2) + ((es[1, 3] - gt[1, 3]) ** 2) + ((es[2, 3] - gt[2, 3]) ** 2)
          for idx, (gt, es) in enumerate(zip(ground_truth[:nframes_est, ...], estimated))]
    return np.array(se).mean()


def drawMatches(img1, img2, kp1, kp2, matches):
    merge_img = cv2.hconcat([img1, img2])
    merge_img = cv2.cvtColor(merge_img, cv2.COLOR_GRAY2BGR)
    for m in matches:
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (b, g, r)
        p1 = kp1[m.queryIdx]
        p2 = kp2[m.trainIdx]

        x1, y1 = map(lambda x: int(round(x)), p1)
        x2, y2 = map(lambda x: int(round(x)), p2)
        cv2.circle(merge_img, (x1, y1), 3, (255))

        cv2.circle(merge_img, (img1.shape[1] + x2, y2), 3, rand_color)
        cv2.line(merge_img, (x1, y1), (img1.shape[1] + x2, y2), rand_color)
    return merge_img


def filter_matches_distance(matches, dist_threshold=0.5):
    filtered_matches = []
    for m, n in matches:
        if m.distance <= dist_threshold * n.distance:
            filtered_matches.append(m)
    return filtered_matches


def match_features(des1, des2, matching='BF', detector='sift', sort=False, k=2):
    if matching == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
    elif matching == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches = matcher.knnMatch(des1, des2, k=k)

    if sort:
        matches = sorted(matches, key=lambda x: x[0].distance)

    return matches


def extract_features(image, detector='sift', GoodP=False, mask=None):
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
    if GoodP:
        pts = cv2.goodFeaturesToTrack(
            image, 3000, qualityLevel=0.01, minDistance=7)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=15) for f in pts]
        kp, des = det.compute(image, kps)

    else:
        kp, des = det.detectAndCompute(image, mask)
    kp = np.array([(k.pt[0], k.pt[1]) for k in kp])
    return kp, des


def calc_depth_map(disp_left, k_left, t_left, t_right, rectified=True):
    if rectified:
        b = t_right[0] - t_left[0]
    else:
        b = t_left[0] - t_right[0]

    f = k_left[0][0]

    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1

    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left

    return depth_map


def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    return k, r, t


def compute_left_disparity_map(img_left, img_right, matcher='bm', verbose=False):
    sad_window = 6
    num_disparities = sad_window * 16
    block_size = 11
    matcher_name = matcher

    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size)

    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1=8 * 3 * block_size ** 2,
                                        P2=32 * 3 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    start = time.perf_counter()
    disp_left = matcher.compute(img_left, img_right).astype(np.float32) / 16
    end = time.perf_counter()

    if verbose:
        print(
            f'Time to compute disparity map using Stereo{matcher_name.upper()}', end - start)

    return disp_left


def viz_3d(pt_3d):
    X = pt_3d[0, :]
    Y = pt_3d[1, :]
    Z = pt_3d[2, :]

    mlab.points3d(
        X,  # x
        Y,  # y
        Z,  # z
        mode="point",  # How to render each point {'point', 'sphere' , 'cube' }
        colormap='copper',  # 'bone', 'copper',
        line_width=10,
        scale_factor=1
    )
    # mlab.axes(xlabel='x', ylabel='y', zlabel='z',ranges=(0,20,0,20,0,10),nb_labels=10)
    mlab.show()


def viz_3d_matplotlib(pt_3d):
    X = pt_3d[0, :]
    Y = pt_3d[1, :]
    Z = pt_3d[2, :]

    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X,
               Y,
               Z,
               s=1,
               cmap='gray')

    plt.show()


def draw_epipolar_lines(pts1, pts2, img1, img2):
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, )
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, )
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.subplot(121)
    plt.imshow(img5)
    plt.subplot(122)
    plt.imshow(img3)
    plt.show()


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def estimate_motion(matches, kp1, kp2, k=None, depth1=None, max_depth=3000):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))

    image1_points = np.float32([kp1[m.queryIdx] for m in matches])
    image2_points = np.float32([kp2[m.trainIdx] for m in matches])
    if depth1 is not None:
        cx = k[0, 2]
        cy = k[1, 2]
        fx = k[0, 0]
        fy = k[1, 1]

        object_points = np.zeros((0, 3))
        delete = []

        for i, (u, v) in enumerate(image1_points):
            z = depth1[int(round(v)), int(round(u))]

            if z > max_depth:
                delete.append(i)
                continue

            x = z * (u - cx) / fx
            y = z * (v - cy) / fy
            object_points = np.vstack([object_points, np.array([x, y, z])])
        image1_points = np.delete(image1_points, delete, 0)
        image2_points = np.delete(image2_points, delete, 0)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image2_points, k, None)
        rmat = cv2.Rodrigues(rvec)[0]
    else:
        # Compute the essential matrix
        if k is not None:
            essential_matrix, mask = cv2.findEssentialMat(image1_points, image2_points, cameraMatrix=k,
                                                          method=cv2.RANSAC,
                                                          prob=0.999, threshold=3.0)
        else:
            essential_matrix, mask = cv2.findEssentialMat(image1_points, image2_points, focal=1.0, pp=(
                0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)

        if mask is not None:
            mask = mask.astype(bool).flatten()
            image1_points, image2_points = image1_points[mask], image2_points[mask]

        # Recover the relative pose of the cameras
        _, rmat, tvec, mask = cv2.recoverPose(
            essential_matrix, image1_points, image2_points, k)
    return rmat, tvec, image1_points, image2_points


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t[:, 0]
    return ret


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret


def triangulatecv(P1, P2, image1_points, image2_points):
    return cv2.triangulatePoints(P1[:3], P2[:3], image1_points.T, image2_points.T).T


def calc_dist(origin, ls):
    dist = ((ls[:, 0] - origin[0]) ** 2 + (ls[:, 1] - origin[1]) ** 2 + (ls[:, 2] - origin[2]) ** 2) ** 0.5
    return dist


def norm_points(img1pts, img2pts, K):
    img1ptsHom = cv2.convertPointsToHomogeneous(img1pts)[:, 0, :]
    img2ptsHom = cv2.convertPointsToHomogeneous(img2pts)[:, 0, :]

    img1ptsNorm = (np.linalg.inv(K).dot(img1ptsHom.T)).T
    img2ptsNorm = (np.linalg.inv(K).dot(img2ptsHom.T)).T

    img1ptsNorm = cv2.convertPointsFromHomogeneous(img1ptsNorm)[:, 0, :]
    img2ptsNorm = cv2.convertPointsFromHomogeneous(img2ptsNorm)[:, 0, :]

    return img1ptsNorm, img2ptsNorm
