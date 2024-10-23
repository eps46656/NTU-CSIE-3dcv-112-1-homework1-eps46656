import sys
import random
import numpy as np
import cv2 as cv
import utils

def get_sift_correspondences(img1, img2, idxes):
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

    if idxes != None:
        good_matches = [good_matches[idx] for idx in idxes]

    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if idxes != None:
        cv.imshow('match', img_draw_match)
        cv.waitKey(0)

    return points1, points2

def CalcErr(points1, points2):
    # points1[P, N]
    # points2[P, N]

    assert len(points1.shape) == 2
    assert len(points2.shape) == 2
    assert points1.shape == points2.shape

    return np.sqrt(((points1 - points2)**2).sum(0))

def Problem(points1, points2, gt_points1, gt_points2, normalized):
    # points1: [P, N]
    # points2: [P, N]
    # gt_points1: [Q, N]
    # gt_points2: [Q, N]
    # normalized: normalized DLT or not

    T1, T2, H = utils.NormalizedDLT(points1, points2, normalized)

    rep_gt_points1 = T1 @ gt_points1
    rep_gt_points2 = T2 @ gt_points2

    rep_pr_points2 = utils.HomographyTrans(H, rep_gt_points1) # [3, M]

    pr_points2 = np.linalg.inv(T2) @ rep_pr_points2

    rep_err = CalcErr(rep_pr_points2[:2, :], rep_gt_points2[:2, :]).mean()
    err = CalcErr(pr_points2[:2, :], gt_points2[:2, :]).mean()

    return rep_err, err

def main():
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = sys.argv[3]
    normalized = sys.argv[4] == "True"
    k = int(sys.argv[5])

    points1, points2 = get_sift_correspondences(img1, img2, None)
    ones = np.ones([1, points1.shape[0]], dtype=points1.dtype)
    points1 = np.concatenate([points1.transpose(), ones], 0) # [3, N]
    points2 = np.concatenate([points2.transpose(), ones], 0) # [3, N]

    gt_points = np.load(gt_correspondences)
    ones = np.ones([1, gt_points.shape[1]], dtype=gt_points.dtype)
    gt_points1 = np.concatenate([gt_points[0].transpose(), ones], 0) # [3, M]
    gt_points2 = np.concatenate([gt_points[1].transpose(), ones], 0) # [3, M]

    progress = False
    best_idxes = []
    best_rep_err = 10**9
    best_err = 10**9

    for _ in range(1024 * 8 * 8):
        idxes = random.choices(range(points1.shape[1]), k=k)
        idxes.sort()

        rep_err, err = Problem(
            points1[:, idxes], points2[:, idxes],
            gt_points1, gt_points2, normalized)

        if err < best_err:
            progress = True
            best_idxes = idxes
            best_rep_err = rep_err
            best_err = err

    print(f"progress = {progress}")
    print(f"\n\n")
    print(f"    best_idxes = {best_idxes}")
    print(f"    best_rep_err = {best_rep_err}")
    print(f"    best_err = {best_err}")

if __name__ == '__main__':
    main()
