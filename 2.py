import sys
import math
import numpy as np
import cv2 as cv
import utils


WINDOW_NAME = 'window'


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[USAGE] python3 mouse_click_example.py [IMAGE PATH]')
        sys.exit(1)

    img = cv.imread(sys.argv[1])
    dst_path = sys.argv[2]
    dst_Homo = sys.argv[3]
    dst_h = int(sys.argv[4])
    dst_w = int(sys.argv[5])

    points_add= []
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, [points_add])
    while True:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exist when pressing ESC

    cv.destroyAllWindows()

    print('{} Points added'.format(len(points_add)))

    assert 4 <= len(points_add)

    points = np.array([
        [points_add[0][1], points_add[1][1], points_add[2][1], points_add[3][1]],
        [points_add[0][0], points_add[1][0], points_add[2][0], points_add[3][0]],
        [1, 1, 1, 1],
    ], dtype=np.float32)

    corners = np.array([
        [0,       0, dst_h-1, dst_h-1],
        [0, dst_w-1, dst_w-1,       0],
        [1,       1,       1,       1],
    ])

    T1 = utils.GetNormalizedMat(corners, [0, 0], math.sqrt(2))
    T2 = utils.GetNormalizedMat(points, [0, 0], math.sqrt(2))

    rep_corners = T1 @ corners
    rep_points = T2 @ points

    H = utils.FindHomographic(
        rep_corners,
        rep_points
    )

    print(f"H =\n{H}")

    np.save(dst_Homo, H, allow_pickle=True)

    idxes = np.mgrid[:dst_h, :dst_w] # [2, dst_h, dst_w]
    idxes = np.concatenate([idxes, np.ones([1, dst_h, dst_w])], 0)
    # [3, dst_h, dst_w]

    idxes = idxes.reshape([3, dst_h*dst_w]) # [3, dst_h*dst_w]

    idxes = utils.HomographyTrans(H, T1 @ idxes)
    # [3, dst_h*dst_w]

    idxes = (np.linalg.inv(T2) @ idxes)[:-1, :] # [2, dst_h*dst_w]
    idxes = idxes.reshape([2, dst_h, dst_w]).transpose([1, 2, 0])
    # [dst_h, dst_w, 2]

    img_dst = utils.BatchSampling(
        img, idxes, utils.BicubicSampling).clip(0, 255).astype(np.uint8)

    cv.imwrite(dst_path, img_dst)

    cv.imshow(WINDOW_NAME, img_dst)
    cv.waitKey(0)
