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
    keypoint_names = [
        "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", "Left Shoulder", "Right Shoulder",
        "Left Elbow", "Right Elbow", "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
        "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
    ]
    for i, name in enumerate(keypoint_names):
        print("{0}\t({1}, {2}, {3})".format(name, points3Ds[i][0], points3Ds[i][1], points3Ds[i][2]))
