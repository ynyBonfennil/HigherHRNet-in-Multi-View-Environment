import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "3rdparty/simple-HigherHRNet/"))

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from scipy.linalg import svd
from SimpleHigherHRNet import SimpleHigherHRNet


def multi_view_triangulation(points2Ds, camera_matrices):
    # variable definitions are based on this article
    # https://mem-archive.com/2018/11/04/post-867/

    # iterate along keypoint definition
    points3Ds = []
    for keynum in range(17):
        A = []
        # iterate along cameras
        for _, (points2D, cammat) in enumerate(zip(points2Ds, camera_matrices)):

            # exclude keypoint with low reliability
            if points2D[keynum][2] < 0.5:
                continue

            u = points2D[keynum][1]
            v = points2D[keynum][0]
            p1 = cammat[0]
            p2 = cammat[1]
            p3 = cammat[2]
            A.append(u*p3 - p1)
            A.append(v*p3 - p2)
        np.array(A)

        # solve Singular Value Decomposition
        svd_result = svd(A, full_matrices=False)

        # Since singular values are sorted in Non-increasing order,
        # svd_result[1][-1] is the smallest of all singular values
        # and corresponding singular vector is svd_result[2][-1]
        ans = svd_result[2][-1] / svd_result[2][-1][3]
        points3Ds.append(ans)
    
    return np.array(points3Ds)


if __name__ == "__main__":
    model = SimpleHigherHRNet(32, 17, "./checkpoints/pose_higher_hrnet_w32_512.pth")
    images = [
        "./sample/images/Camera.001.png",
        "./sample/images/Camera.002.png",
        "./sample/images/Camera.003.png",
        "./sample/images/Camera.004.png",
        "./sample/images/Camera.005.png",
    ]
    calibs = [
        "./sample/calibrations/Camera.001.xml",
        "./sample/calibrations/Camera.002.xml",
        "./sample/calibrations/Camera.003.xml",
        "./sample/calibrations/Camera.004.xml",
        "./sample/calibrations/Camera.005.xml",
    ]

    # estimate keypoints in 2D
    points2Ds = []
    for item in images:
        image = cv2.imread(item)
        joints = model.predict(image)
        points2Ds.append(joints[0])

    # load camera parameters
    cammat = []
    for item in calibs:
        tree = ET.parse(item)
        intrinsics = np.array([float(val) for val in tree.getroot()[1][3].text.split()]).reshape([3, 3])
        extrinsics = np.array([float(val) for val in tree.getroot()[0][3].text.split()]).reshape([3, 4])
        cammat.append(intrinsics @ extrinsics)
    
    # estimate keypoints in 3d
    points3Ds = multi_view_triangulation(points2Ds, cammat)
    points3Ds = np.round(points3Ds, decimals=5)

    # print
    print("Nose: ({0}, {1}, {2})".format(points3Ds[0][0], points3Ds[0][1], points3Ds[0][2]))
    print("Left Eye: ({0}, {1}, {2})".format(points3Ds[1][0], points3Ds[1][1], points3Ds[1][2]))
    print("Right Eye: ({0}, {1}, {2})".format(points3Ds[2][0], points3Ds[2][1], points3Ds[2][2]))
    print("Left Ear: ({0}, {1}, {2})".format(points3Ds[3][0], points3Ds[3][1], points3Ds[3][2]))
    print("Right Ear: ({0}, {1}, {2})".format(points3Ds[4][0], points3Ds[4][1], points3Ds[4][2]))
    print("Left Shoulder: ({0}, {1}, {2})".format(points3Ds[5][0], points3Ds[5][1], points3Ds[5][2]))
    print("Right Shoulder: ({0}, {1}, {2})".format(points3Ds[6][0], points3Ds[6][1], points3Ds[6][2]))
    print("Left Elbow: ({0}, {1}, {2})".format(points3Ds[7][0], points3Ds[7][1], points3Ds[7][2]))
    print("Right Elbow: ({0}, {1}, {2})".format(points3Ds[8][0], points3Ds[8][1], points3Ds[8][2]))
    print("Left Wrist: ({0}, {1}, {2})".format(points3Ds[9][0], points3Ds[9][1], points3Ds[9][2]))
    print("Right Wrist: ({0}, {1}, {2})".format(points3Ds[10][0], points3Ds[10][1], points3Ds[10][2]))
    print("Left Hip: ({0}, {1}, {2})".format(points3Ds[11][0], points3Ds[11][1], points3Ds[11][2]))
    print("Right Hip: ({0}, {1}, {2})".format(points3Ds[12][0], points3Ds[12][1], points3Ds[12][2]))
    print("Left Knee: ({0}, {1}, {2})".format(points3Ds[13][0], points3Ds[13][1], points3Ds[13][2]))
    print("Right Knee: ({0}, {1}, {2})".format(points3Ds[14][0], points3Ds[14][1], points3Ds[14][2]))
    print("Left Ankle: ({0}, {1}, {2})".format(points3Ds[15][0], points3Ds[15][1], points3Ds[15][2]))
    print("Right Ankle: ({0}, {1}, {2})".format(points3Ds[16][0], points3Ds[16][1], points3Ds[16][2]))
