import sys
import math
import random
import numpy as np
import cv2 as cv
import utils

def get_sift_correspondences(img1, img2, k):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # good_matches = random.choices(
    #   sorted(good_matches, key=lambda x: x.distance), k=k)

    good_matches = random.choices(good_matches, k=k)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    # img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # cv.imshow('match', img_draw_match)
    # cv.waitKey(0)
    return points1, points2

def Problem(img1, img2, gt_correspondences, k, normalized):
    # img1: [H, W, C]
    # img2: [H, W, C]
    # gt_correspondences: filename
    # normalized: normalized DLT or not

    points1, points2 = get_sift_correspondences(img1, img2, k)
    N = points1.shape[0]
    ones = np.ones([1, N], dtype=points1.dtype)
    points1 = np.concatenate([points1.transpose(), ones], 0) # [3, N]
    points2 = np.concatenate([points2.transpose(), ones], 0) # [3, N]

    gt_points = np.load(gt_correspondences)
    M = gt_points.shape[1]
    ones = np.ones([1, M], dtype=gt_points.dtype)
    gt_points1 = np.concatenate([gt_points[0].transpose(), ones], 0) # [3, M]
    gt_points2 = np.concatenate([gt_points[1].transpose(), ones], 0) # [3, M]

    T1 = np.diag([1, 1, 1])
    T2 = np.diag([1, 1, 1])

    if normalized:
        T1 = utils.GetNormalizedMat(points1, [0, 0], math.sqrt(2))
        T2 = utils.GetNormalizedMat(points2, [0, 0], math.sqrt(2))

    T2_inv = np.linalg.inv(T2)

    rep_points1 = T1 @ points1
    rep_points2 = T2 @ points2

    rep_gt_points1 = T1 @ gt_points1
    rep_gt_points2 = T2 @ gt_points2

    H = utils.FindHomographic(rep_points1, rep_points2) # [3, 3]

    rep_pr_points2 = H @ rep_gt_points1 # [3, M]
    np.divide(rep_pr_points2[0], rep_pr_points2[2], out=rep_pr_points2[0])
    np.divide(rep_pr_points2[1], rep_pr_points2[2], out=rep_pr_points2[1])
    np.divide(rep_pr_points2[2], rep_pr_points2[2], out=rep_pr_points2[2])

    pr_points2 = T2_inv @ rep_pr_points2

    diff = rep_pr_points2[0:2, :] - rep_gt_points2[0:2, :]
    rep_err = np.sqrt((diff**2).sum(0)).mean()

    diff = pr_points2[0:2, :] - gt_points2[0:2, :]
    err = np.sqrt((diff**2).sum(0)).mean()

    return rep_err, err

def main(img1_path, img2_path, gt_correspondences, k, normalized):
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    Problem(img1, img2, gt_correspondences, k, normalized)


if __name__ == '__main__':
    img1 = cv.imread("images/1-0.png")
    img2 = cv.imread("images/1-2.png")
    gt_correspondences = "groundtruth_correspondences/correspondence_02.npy"
    k = 20
    normalized = False

    test_num = 64

    acc_rep_err = 0.0
    acc_err = 0.0

    for i in range(test_num):
        print(f"test iteration: {i}")

        rep_err, err = Problem(img1, img2, gt_correspondences, k, normalized)

        acc_rep_err += rep_err
        acc_err += err

    avg_rep_err = acc_rep_err / test_num
    avg_err = acc_err / test_num

    print(f"acc_rep_err: {acc_rep_err}")
    print(f"acc_err: {acc_err}")
    print(f"avg_rep_err: {avg_rep_err}")
    print(f"avg_err: {avg_err}")
