import numpy as np
import cv2
import matplotlib

import matplotlib.pyplot as plt
# def flow2color(flow):
#     """
#     convert optical flow to rgb
#     opencv color wheel: https://i.imgur.com/PKjgfFXm.jpg
#     Parameters
#     ----------
#     flow : (Height, width, 2) array,
#         optical flow
#
#     return
#     ----------
#     rgb : (Height, width, 3) array
#     """
#     hsv = np.zeros(flow.shape[:2] + (3, )).astype(np.uint8)
#     hsv[..., 1] = 255
#     mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
#     hsv[..., 0] = ang*180/np.pi/2
#     hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
#     rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#     return rgb
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

def draw_arrow(image, p, q, color, arrow_magnitude=9, thickness=1, line_type=8, shift=0):
    # adapted from http://mlikihazar.blogspot.com.au/2013/02/draw-arrow-opencv.html

    # draw arrow tail
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # calc angle of the arrow
    angle = np.arctan2(p[1]-q[1], p[0]-q[0])
    # starting point of first line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle + np.pi/4)),
         int(q[1] + arrow_magnitude * np.sin(angle + np.pi/4)))
    # draw first half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)
    # starting point of second line of arrow head
    p = (int(q[0] + arrow_magnitude * np.cos(angle - np.pi/4)),
         int(q[1] + arrow_magnitude * np.sin(angle - np.pi/4)))
    # draw second half of arrow head
    cv2.line(image, p, q, color, thickness, line_type, shift)

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow2color(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    # maxu = max(maxu, np.max(u))
    # minu = min(minu, np.min(u))
    #
    # maxv = max(maxv, np.max(v))
    # minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    # print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def flow2vector(flow, interval=30):
    """
    convert optical flow to velocity vector
    Parameters
    ----------
    flow : (Height, width, 2) array,
        optical flow

    return
    ----------
    rgb : (Height, width, 3) array
    """
    rgb = np.ones(flow.shape[:2]+(3,), dtype=np.uint8)
    for i in range(10, rgb.shape[0]-10, interval):
        for j in range(10, rgb.shape[1]-10, interval):
            # ignore NaN
            if flow[i,j,0] == flow[i,j,0]:
                try:
                    # opencv 3.1.0
                    if flow.shape[-1] == 2:
                        cv2.arrowedLine(rgb,
                                        (j, i),
                                        (j + int(round(flow[i, j, 0])), i + int(round(flow[i, j, 1]))),
                                        (150, 0, 0), 2)
                    else:
                        cv2.arrowedLine(rgb, (j, i), (j + int(round(flow[i, j, 0])), i), (150, 0, 0), 2)
                except AttributeError:
                    # opencv 2.4.8


                    if flow.shape[-1] == 2:
                        if np.power(np.power(flow[i,j],2).sum(),0.5) < 0.3:
                            continue
                        draw_arrow(rgb, (j, i),
                                   (j + int(round(flow[i, j, 0])),
                                    i + int(round(flow[i, j, 1]))),
                                   color=(200, 150, 150),
                                   arrow_magnitude=2,
                                   thickness=1,
                                   line_type=4,
                                   shift=0)
                    else:
                        if np.abs(flow[i, j]) < 0.3:
                            continue
                        draw_arrow(rgb,
                                   (j, i),
                                   (j + int(round(flow[i, j])),
                                    i),
                                   color=(150, 0, 0),
                                   arrow_magnitude=2,
                                   thickness=1,
                                   line_type=4,
                                   shift=0)

    return rgb

def check_data(img1, img2, type, gt, interval=10, number=20, y_begin=100, x_begin=100, waitforkey=True):
    """
    check the validity of ground truth
    Parameters
    ----------
    img1 : (height, width, 3) array
        left image
    img2 : (height, width, 3) array
        right image
    type: string, 'stereo' or 'flow'
        indicate data type
    gt : array,
        ground truth,
        if type is 'stereo', gt should be (height, width) ndarray, otherwise gt should be (height, width, 2) ndarray
    interval : int
        interval between adjacent plots
    number : int
        total number of plots
    y_begin : int
    x_begin : int
    """
    tot = 0
    for i in range(y_begin, img1.shape[0], interval):
        for j in range(x_begin, img1.shape[1], interval):
            if tot>number:
                break
            if type == 'stereo' :
                # NaN
                if gt[i,j]!=gt[i,j]:
                    continue
                plt.figure()
                plt.imshow(img1[i - 15:i + 16, j - 15:j + 16])
                plt.title('patch in img1 x : {} y: {}'.format(j, i))
                if waitforkey:
                    plt.waitforbuttonpress()
                plt.figure()
                plt.imshow(img2[i - 15:i + 16, j - int(round(gt[i, j])) - 15:j + 16 - int(round(gt[i, j]))])
                plt.title('corresponding patch in img2 x : {} y: {}'.format(j+gt[i,j], i))
                if waitforkey:
                    plt.waitforbuttonpress()
                tot += 1
            elif type == 'flow':
                # NaN
                if gt[i,j,0]!=gt[i,j,0]:
                    continue
                plt.figure()
                plt.imshow(img1[i - 15:i + 16, j - 15:j + 16])
                plt.title('patch in img1 x : {} y : {}'.format(j, i))
                if waitforkey:
                    plt.waitforbuttonpress()
                plt.figure()
                plt.imshow(img2[i - 15 + int(round(gt[i, j, 1])):i + 16 + int(round(gt[i, j, 1])),
                           j + int(round(gt[i, j, 0])) - 15:j + 16 + int(round(gt[i, j, 0]))])
                plt.title('corresponding patch in img2 x: {} y: {}'.format(j + gt[i, j, 0], i + gt[i, j, 1]))
                if waitforkey:
                    plt.waitforbuttonpress()
                tot += 1

def plot(img, name, waitforkey=True):

    plt.figure()
    plt.imshow(img)
    plt.title(name)
    plt.colorbar()
    if waitforkey:
        plt.waitforbuttonpress()

def plot_pairs(img1, img2, label, type, plot_patch=True, waitforkey=True, interval=20, num=10, y_begin=100, x_begin=100):

    plot(img1, 'img1', waitforkey)
    plot(img2, 'img2', waitforkey)
    if type == 'stereo':
        plot(label, 'disparity', waitforkey)
    elif type == 'flow':
        color = flow2color(label)
        vector = flow2vector(label)
        plot(color, 'color', waitforkey)
        plot(vector, 'vector', waitforkey)

    if plot_patch:
        check_data(img1, img2, type, label, interval=interval, number=num, y_begin=y_begin, x_begin=x_begin, waitforkey=waitforkey)
