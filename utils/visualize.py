import numpy as np
import cv2
import matplotlib.pyplot as plt

def flow2color(flow):
    """
    convert optical flow to rgb
    opencv color wheel: https://i.imgur.com/PKjgfFXm.jpg
    Parameters
    ----------
    flow : (Height, width, 2) array,
        optical flow

    return
    ----------
    rgb : (Height, width, 3) array
    """
    hsv = np.zeros(flow.shape[:2] + (3, )).astype(np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rgb

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
    rgb = np.ones(flow.shape[:2]+(3,))
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
                        cv2.line(rgb,
                                 (j, i),
                                 (j + int(round(flow[i, j, 0])),
                                  i + int(round(flow[i, j, 1]))),
                                 (150, 0, 0), 2)
                    else:
                        cv2.line(rgb,
                                 pt1 = (j, i),
                                 pt2 = (j + int(round(flow[i, j])), i),
                                 color = (150, 0, 0),
                                 thickness =  1)

    return rgb

def check_data(img1, img2, type, gt, interval=10, number=20, y_begin=100, x_begin=100):
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
                plt.waitforbuttonpress()
                plt.figure()
                plt.imshow(img2[i - 15:i + 16, j - int(round(gt[i, j])) - 15:j + 16 - int(round(gt[i, j]))])
                plt.title('corresponding patch in img2 x : {} y: {}'.format(j+gt[i,j], i))
                plt.waitforbuttonpress()
                tot += 1
            elif type == 'flow':
                # NaN
                if gt[i,j,0]!=gt[i,j,0]:
                    continue
                plt.figure()
                plt.imshow(img1[i - 15:i + 16, j - 15:j + 16])
                plt.title('patch in img1 x : {} y : {}'.format(j, i))
                plt.waitforbuttonpress()
                plt.figure()
                plt.imshow(img2[i - 15 + int(round(gt[i, j, 1])):i + 16 + int(round(gt[i, j, 1])),
                           j + int(round(gt[i, j, 0])) - 15:j + 16 + int(round(gt[i, j, 0]))])
                plt.title('corresponding patch in img2 x: {} y: {}'.format( i + gt[i, j, 0], j + gt[i, j, 1]))
                plt.waitforbuttonpress()
                tot += 1

def plot(img, name):

    plt.figure()
    plt.imshow(img)
    plt.title(name)
    plt.waitforbuttonpress()

def plot_pairs(img1, img2, label, type):

    plot(img1, 'img1')
    plot(img2, 'img2')
    if type == 'stereo':
        plot(label, 'disparity')
    elif type == 'flow':
        color = flow2color(label)
        vector = flow2vector(label)
        plot(color, 'color')
        plot(vector, 'vector')
    check_data(img1, img2, type, label, interval=20, number=2, y_begin=100, x_begin=100)
