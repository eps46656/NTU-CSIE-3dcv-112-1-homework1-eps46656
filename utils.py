import math
import numpy as np
import cv2 as cv

def LinearInterp(data, points):
    # data[B, 2]
    # points[B]

    B = data.shape[0]

    assert data.shape == (B, 2)
    assert points.shape == (B,)

    return data[:, 1] * points + data[:, 0] * (1 - points)

def CubicInterpBasic(data, points):
    # data[B, 4]
    # points[B]

    B = data.shape[0]

    assert data.shape == (B, 4)
    assert points.shape == (B,)

    x0 = -1
    x1 =  0
    x2 =  1
    x3 =  2

    M = np.linalg.inv(np.array([
        [x0**0, x1**0, x2**0, x3**0],
        [x0**1, x1**1, x2**1, x3**1],
        [x0**2, x1**2, x2**2, x3**2],
        [x0**3, x1**3, x2**3, x3**3],
    ], dtype=data.dtype)) # [4, 4]

    ret = np.empty([B], dtype=data.dtype)

    for b in range(B):
        coeffs = data[b] @ M

        p = points[b]

        cur = coeffs[3]
        cur = cur * p + coeffs[2]
        cur = cur * p + coeffs[1]
        cur = cur * p + coeffs[0]

        ret[b] = cur

    return ret

def CubicInterp(data, points):
    # data[B, 4]
    # points[B]

    B = data.shape[0]

    assert data.shape == (B, 4)
    assert points.shape == (B,)

    x0 = -1
    x1 =  0
    x2 =  1
    x3 =  2

    M = np.linalg.inv(np.array([
        [x0**0, x1**0, x2**0, x3**0],
        [x0**1, x1**1, x2**1, x3**1],
        [x0**2, x1**2, x2**2, x3**2],
        [x0**3, x1**3, x2**3, x3**3],
    ], dtype=data.dtype)) # [4, 4]

    ret = data @ M[:, 3]
    np.multiply(ret, points, out=ret)
    np.add(ret, data @ M[:, 2], out=ret)
    np.multiply(ret, points, out=ret)
    np.add(ret, data @ M[:, 1], out=ret)
    np.multiply(ret, points, out=ret)
    np.add(ret, data @ M[:, 0], out=ret)

    return ret

def BicubicSamplingBasic(data, points):
    # data[H, W, ...]
    # points[..., 2]

    assert 2 <= len(data.shape)
    assert points.shape[-1] == 2

    original_data_shape = data.shape
    original_points_shape = points.shape

    H, W = data.shape[:2]

    data = data.reshape(H, W, -1)
    points = points.reshape(-1, 2)

    C = data.shape[2]
    N = points.shape[0]

    ret = np.empty([N, C], dtype=data.dtype)

    for n in range(N):
        h = points[n][0]
        w = points[n][1]

        idx_h = np.floor(h).astype(np.int32)
        idx_w = np.floor(w).astype(np.int32)

        res_h = h - idx_h
        res_w = w - idx_w

        ds = np.empty([4, 4, C], dtype=data.dtype)

        for i in range(4):
            for j in range(4):
                ds[i][j] = data[(idx_h + (i - 1)).clip(0, H-1),
                                (idx_w + (j - 1)).clip(0, W-1)]

        for c in range(C):
            cur = np.empty([1, 4], dtype=data.dtype)

            for i in range(4):
                cur[0][i] = CubicInterp(
                    np.array([[
                        ds[i][0][c], ds[i][1][c], ds[i][2][c], ds[i][3][c],]]),
                    np.array([res_w]))[0]

            ret[n, c] = CubicInterp(cur, np.array([res_h]))[0]

    return ret.reshape(
        list(original_points_shape)[:-1] + list(original_data_shape)[2:])

def BilinearSampling(data, points, out=None):
    # data[H, W, ...]
    # points[..., 2]

    assert 2 <= len(data.shape)
    assert points.shape[-1] == 2

    original_data_shape = data.shape
    original_points_shape = points.shape

    H, W = data.shape[:2]

    C_shape = original_data_shape[2:]
    N_shape = original_points_shape[:-1]

    if out is not None:
        assert out.shape == N_shape + C_shape

    data = data.reshape(H, W, -1)
    points = points.reshape(-1, 2)

    C = data.shape[2]
    N = points.shape[0]

    points_idx_h = np.floor(points[:, 0]).astype(dtype=np.int32)
    points_idx_w = np.floor(points[:, 1]).astype(dtype=np.int32)

    points_res_h = points[:, 0] - points_idx_h # [N]
    points_res_w = points[:, 1] - points_idx_w # [N]

    ds = np.empty([2, 2, N, C], dtype=data.dtype)

    for i in range(2):
        for j in range(2):
            ds[i][j] = data[(points_idx_h + i).clip(0, H-1),
                            (points_idx_w + j).clip(0, W-1)]

    del points_idx_h
    del points_idx_w

    ret = np.empty([N, C], dtype=data.dtype) if out is None else out.reshape([N, C])

    for c in range(C):
        cur = np.empty([N, 2], dtype=data.dtype)

        cur[:, 0] = LinearInterp(ds[0, :, :, c].transpose([1, 0]), points_res_w)
        cur[:, 1] = LinearInterp(ds[1, :, :, c].transpose([1, 0]), points_res_w)

        ret[:, c] = LinearInterp(cur, points_res_h)

    return ret.reshape(N_shape + C_shape)

def BicubicSampling(data, points, out=None):
    # data[H, W, ...]
    # points[..., 2]

    assert 2 <= len(data.shape)
    assert points.shape[-1] == 2

    original_data_shape = data.shape
    original_points_shape = points.shape

    H, W = data.shape[:2]

    C_shape = original_data_shape[2:]
    N_shape = original_points_shape[:-1]

    if out is not None:
        assert out.shape == N_shape + C_shape

    data = data.reshape(H, W, -1)
    points = points.reshape(-1, 2)

    C = data.shape[2]
    N = points.shape[0]

    points_idx_h = np.floor(points[:, 0]).astype(np.int32) # [N]
    points_idx_w = np.floor(points[:, 1]).astype(np.int32) # [N]

    points_res_h = points[:, 0] - points_idx_h # [N]
    points_res_w = points[:, 1] - points_idx_w # [N]

    ds = np.empty([4, 4, N, C], dtype=data.dtype)

    for i in range(4):
        for j in range(4):
            ds[i][j] = data[(points_idx_h + (i - 1)).clip(0, H-1),
                            (points_idx_w + (j - 1)).clip(0, W-1)]

    del points_idx_h
    del points_idx_w

    ret = np.empty([N, C], dtype=data.dtype) if out is None else out.reshape([N, C])

    for c in range(C):
        cur = np.empty([N, 4], dtype=data.dtype)

        cur[:, 0] = CubicInterp(ds[0, :, :, c].transpose([1, 0]), points_res_w)
        cur[:, 1] = CubicInterp(ds[1, :, :, c].transpose([1, 0]), points_res_w)
        cur[:, 2] = CubicInterp(ds[2, :, :, c].transpose([1, 0]), points_res_w)
        cur[:, 3] = CubicInterp(ds[3, :, :, c].transpose([1, 0]), points_res_w)

        ret[:, c] = CubicInterp(cur, points_res_h)

    return ret.reshape(N_shape + C_shape)

def BatchSampling(data, points, sampler, out=None):
    # data[H, W, ...]
    # points[..., 2]

    assert 2 <= len(data.shape)
    assert points.shape[-1] == 2

    original_data_shape = data.shape
    original_points_shape = points.shape

    H, W = data.shape[:2]

    C_shape = original_data_shape[2:]
    N_shape = original_points_shape[:-1]

    if out is not None:
        assert out.shape == N_shape + C_shape

    data = data.reshape(H, W, -1)
    points = points.reshape(-1, 2)

    C = data.shape[2]
    N = points.shape[0]

    ret = np.empty([N, C], dtype=data.dtype) if out is None else out.reshape([N, C])

    BAtCH_SIZE = 2**16

    for i in range(0, N, BAtCH_SIZE):
        j = min(N, i + BAtCH_SIZE)
        sampler(data, points[i:j, :], out=ret[i:j, :])

    return ret.reshape(N_shape + C_shape)

def GetNormalizedMat(points, center, dist):
    # points[P, N]
    # center[P-1]
    # dist

    center = np.array(center)

    assert len(points.shape) == 2
    assert points.shape[0] - 1 == center.shape[0]

    P = points.shape[0]

    origin = points.mean(1)[:-1]

    odist = (((points[:-1, :] - origin.reshape([P-1, 1]))**2).sum(0)**0.5).mean()

    k = dist / odist

    ret = np.zeros([P, P])

    for i in range(P-1):
        ret[i, i] = k

    ret[-1, -1] = 1

    ret[:-1, -1] = center - origin * k

    return ret

def FindHomographic(src, dst):
    # src[P, N]
    # dst[Q, N]

    assert len(src.shape) == 2
    assert len(dst.shape) == 2
    assert src.shape[1] == dst.shape[1]

    P, N = src.shape
    Q, _ = dst.shape

    A = np.zeros([N*(Q-1), P*Q])

    for i in range(N):
        for j in range(Q-1):
            A[(Q-1)*i+j, j*P:j*P+P] = src[:, i]
            A[(Q-1)*i+j, -P:] = src[:, i] * -dst[j, i]

    _, _, Vh = np.linalg.svd(A)

    return Vh[-1, :].reshape([Q, P])

def HomographyTrans(H, x):
    assert len(H.shape) == 2
    assert len(x.shape) == 2

    ret = H @ x

    P = ret.shape[0]

    for i in range(P):
        np.divide(ret[i, :], ret[-1, :], out=ret[i, :])

    return ret

def NormalizedDLT(points1, points2, normalized):
    # points1[P, N]
    # points2[Q, N]

    assert len(points1.shape) == 2
    assert len(points2.shape) == 2

    P = points1.shape[0]
    Q = points2.shape[0]

    T1 = np.identity(P)
    T2 = np.identity(Q)

    if normalized:
        T1 = GetNormalizedMat(points1, np.zeros([P-1]), np.sqrt(P-1))
        T2 = GetNormalizedMat(points2, np.zeros([Q-1]), np.sqrt(Q-1))

    rep_points1 = T1 @ points1
    rep_points2 = T2 @ points2

    H = FindHomographic(rep_points1, rep_points2) # [P, Q]

    return T1, T2, H
    # return np.linalg.inv(T2) @ H @ T1
