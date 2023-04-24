import numpy as np
import cv2
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from random import randint

def visualize_trajectory(trajectory):
    # Unpack X Y Z each trajectory point
    locX = []
    locY = []
    locZ = []
    # This values are required for keeping equal scale on each plot.
    # matplotlib equal axis may be somewhat confusing in some situations because of its various scale on
    # different axis on multiple plots
    max = -math.inf
    min = math.inf

    # Needed for better visualisation
    maxY = -math.inf
    minY = math.inf

    for i, tr in enumerate(trajectory):
        current_pos = tr[:, 3]
        locX.append(current_pos.item(0))
        locY.append(current_pos.item(1))
        locZ.append(current_pos.item(2))
        if np.amax(current_pos) > max:
            max = np.amax(current_pos)
        if np.amin(current_pos) < min:
            min = np.amin(current_pos)

        if current_pos.item(1) > maxY:
            maxY = current_pos.item(1)
        if current_pos.item(1) < minY:
            minY = current_pos.item(1)

    auxY_line = locY[0] + locY[-1]
    if max > 0 and min > 0:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2
    elif max < 0 and min < 0:
        minY = auxY_line + (min - max) / 2
        maxY = auxY_line - (min - max) / 2
    else:
        minY = auxY_line - (max - min) / 2
        maxY = auxY_line + (max - min) / 2

    # Set styles
    mpl.rc("figure", facecolor="white")
    plt.style.use("seaborn-whitegrid")

    # Plot the figure
    fig = plt.figure(figsize=(8, 6), dpi=100)
    # fig.canvas.manager.window.wm_geometry("+0+0")

    gspec = gridspec.GridSpec(3, 3)
    ZY_plt = plt.subplot(gspec[0, 1:])
    YX_plt = plt.subplot(gspec[1:, 0])
    traj_main_plt = plt.subplot(gspec[1:, 1:])
    D3_plt = plt.subplot(gspec[0, 0], projection='3d')

    # Actual trajectory plotting ZX
    toffset = 1.06
    traj_main_plt.set_title("Autonomous vehicle trajectory (Z, X)", y=toffset)
    traj_main_plt.set_title("Trajectory (Z, X)", y=1)
    traj_main_plt.plot(locZ, locX, ".-", label="Trajectory",
                       zorder=1, linewidth=1, markersize=4)
    traj_main_plt.set_xlabel("Z")
    # traj_main_plt.axes.yaxis.set_ticklabels([])
    # Plot reference lines
    traj_main_plt.plot([locZ[0], locZ[-1]], [locX[0], locX[-1]],
                       "--", label="Auxiliary line", zorder=0, linewidth=1)
    # Plot camera initial location
    traj_main_plt.scatter([0], [0], s=8, c="red",
                          label="Start location", zorder=2)
    traj_main_plt.set_xlim([min, max])
    traj_main_plt.set_ylim([min, max])
    traj_main_plt.legend(loc=1, title="Legend",
                         borderaxespad=0., fontsize="medium", frameon=True)

    # Plot ZY
    # ZY_plt.set_title("Z Y", y=toffset)
    ZY_plt.set_ylabel("Y", labelpad=-4)
    ZY_plt.axes.xaxis.set_ticklabels([])
    ZY_plt.plot(locZ, locY, ".-", linewidth=1, markersize=4, zorder=0)
    ZY_plt.plot([locZ[0], locZ[-1]], [(locY[0] + locY[-1]) / 2,
                (locY[0] + locY[-1]) / 2], "--", linewidth=1, zorder=1)
    ZY_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    ZY_plt.set_xlim([min, max])
    ZY_plt.set_ylim([minY, maxY])

    # Plot YX
    # YX_plt.set_title("Y X", y=toffset)
    YX_plt.set_ylabel("X")
    YX_plt.set_xlabel("Y")
    YX_plt.plot(locY, locX, ".-", linewidth=1, markersize=4, zorder=0)
    YX_plt.plot([(locY[0] + locY[-1]) / 2, (locY[0] + locY[-1]) / 2],
                [locX[0], locX[-1]], "--", linewidth=1, zorder=1)
    YX_plt.scatter([0], [0], s=8, c="red", label="Start location", zorder=2)
    YX_plt.set_xlim([minY, maxY])
    YX_plt.set_ylim([min, max])

    # Plot 3D
    D3_plt.set_title("3D trajectory", y=toffset)
    D3_plt.plot3D(locX, locZ, locY, zorder=0)
    D3_plt.scatter(0, 0, 0, s=8, c="red", zorder=1)
    D3_plt.set_xlim3d(min, max)
    D3_plt.set_ylim3d(min, max)
    D3_plt.set_zlim3d(min, max)
    D3_plt.tick_params(direction='out', pad=-2)
    D3_plt.set_xlabel("X", labelpad=0)
    D3_plt.set_ylabel("Z", labelpad=0)
    D3_plt.set_zlabel("Y", labelpad=-2)

    # plt.axis('equal')
    D3_plt.view_init(45, azim=30)
    plt.tight_layout()
    plt.show()


def mse(ground_truth, estimated):
    nframes_est = estimated.shape[0]

    se = [((es[0, 3] - gt[0, 3])**2)+((es[1, 3] - gt[1, 3])**2)+((es[2, 3] - gt[2, 3])**2)
          for idx, (gt, es) in enumerate(zip(ground_truth[:nframes_est, ...], estimated))]
    return np.array(se).mean()


def drawMatches(img1, img2, kp1, kp2, matches):
    merge_img = cv2.hconcat([img1, img2])
    merge_img=cv2.cvtColor(merge_img,cv2.COLOR_GRAY2BGR)
    for m in matches:
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (b,g,r)
        p1 = kp1[m.queryIdx]
        p2 = kp2[m.trainIdx]

        x1, y1 = map(lambda x: int(round(x)), p1)
        x2, y2 = map(lambda x: int(round(x)), p2)
        cv2.circle(merge_img, (x1, y1), 3, (255))

        cv2.circle(merge_img, (img1.shape[1]+x2, y2), 3,rand_color)
        cv2.line(merge_img, (x1, y1), (img1.shape[1]+x2, y2), rand_color)
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
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
    end = time.perf_counter()

    if verbose:
        print(
            f'Time to compute disparity map using Stereo{matcher_name.upper()}', end-start)

    return disp_left


def estimate_motion(matches, kp1, kp2, k, depth1=None, max_depth=3000):

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
        essential_matrix, mask = cv2.findEssentialMat(image1_points, image2_points, focal=1.0, pp=(
            0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)

        # Recover the relative pose of the cameras
        _, rmat, tvec, mask = cv2.recoverPose(
            essential_matrix, image1_points, image2_points)
    return rmat, tvec, image1_points, image2_points


prev_acc = np.zeros((3,))
prev_vel = np.zeros((3,))
prev_pos = np.zeros((3,))

def estimate_jerk(position,elapsed_time):
    global prev_acc, prev_vel ,prev_pos
            # Calculate velocity for each axis using previous position data
    vel = (position-prev_pos) / elapsed_time
    
    # # Calculate acceleration for each axis using previous velocity data
    acc = (vel - prev_vel) / elapsed_time

    # # Calculate jerk for each axis using previous acceleration data
    jerk = (acc - prev_acc) / elapsed_time
    prev_pos=position 
    prev_vel = vel
    prev_acc = acc
    
    # print(f'jerk estimation: {jerk_mag} m/s^3 ') 
    jerk = np.nan_to_num(jerk, nan=0)
    return vel,acc,jerk
     
