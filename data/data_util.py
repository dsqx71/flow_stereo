import re
import cv2
import numpy as np
from random import randint
from ..cython import util_cython

def readPFM(file):
    """
    read .PFM file
    Parameters
    ----------
    file : str
        file dir

    Returns
    -------
    data : numpy array
    scale : float
    """
    file = open(file, 'rb')
    header = file.readline().rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)

    return data, scale


def writePFM(file, image, scale=1):
    """
    write .PFM file
    Parameters
    ----------
    file : str
        output dir
    image : numpy array
    scale : float
    """
    file = open(file, 'wb')
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')
    image = np.flipud(image)
    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))
    endian = image.dtype.byteorder
    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale
    file.write('%f\n' % scale)
    image.tofile(file)


def readFLO(file):
    """
    read optical flow from .flo file
    Parameters
    ----------
    file : str
        dir

    Returns
    -------
    data : (Height, Width, 2) numpy array
    """

    tag_float = 202021.25
    with open(file) as f:
        nbands = 2
        tag = np.fromfile(f, np.float32, 1)[0]

        if tag != tag_float:
            raise ValueError('wrong tag possibly due to big-endian machine?')

        width = np.fromfile(f, np.int32, 1)[0]
        height = np.fromfile(f, np.int32, 1)[0]

        tmp = np.fromfile(f, np.float32)
        tmp = tmp.reshape(height, width * nbands)

        flow = np.zeros((height, width, 2))
        flow[:, :, 0] = tmp[:, 0::2]
        flow[:, :, 1] = tmp[:, 1::2]

    return flow

def writeKittiSubmission(data, prefix, index, type='stereo'):
    """
    convert format to meet kitti submission requirement
    Parameters
    ----------
    data : numpy array
        disparity or optical flow
    prefix : str
        output prefix
    index : int
    type : str
        'stereo' or 'flow'
    """
    if type=='stereo':
        data = data*256.0
        data[data<1.0] = 1.0
        data = data.astype(np.uint16)
        cv2.imwrite(prefix+'/%06d_10.png' % index, data)
    elif type=='flow':
        # TODO: I didn't check this part
        #  please refer to io_flow.h in KITTI 2015 development KIT, if you have any questions
        # in BGR order
        flow = np.zeros(data.shape[:2]+(3,))
        flow[:, :, 1] = data[:, :, 1]
        flow[:, :, 2] = data[:, :, 0]
        flow[:, :, 1:] = flow[:, :, 1:] * 64.0 + 32768.0
        flow[flow<0.0] = 0.0
        flow[flow>65535] = 65535
        flow[:, :, 0] = 1
        flow = flow.astype(np.uint16)
        cv2.imwrite(prefix + '/%06d_10.png' % index, flow)


def crop(img1, img2, label, target_height, target_width):
    """
    Crop img1, img2, and label simultaneously

    Parameters
    ----------
    img1 : (height, width, 3) numpy array
    img2 : (height, width, 3) numpy array
    label : numpy array or None
        disparity map or optical flow field
    target_height : int
    target_width : int

    Returns
    -------
    img1_cropped, img2_cropped, label_cropped,
    """
    y_ori, x_ori = img1.shape[:2]

    assert img1.shape == img2.shape, 'inconsistant shape'
    assert img1.shape[:2] == label.shape[:2], 'inconsistant shape'
    assert target_height<=y_ori and target_width<=x_ori, 'wrong target shape'

    # cropping
    x_begin = randint(0, x_ori - target_width)
    y_begin = randint(0, y_ori - target_height)
    if label is not None:
        return img1[y_begin:y_begin+target_height, x_begin:x_begin+target_width], \
               img2[y_begin:y_begin+target_height, x_begin:x_begin+target_width], \
               label[y_begin:y_begin+target_height, x_begin:x_begin+target_width]
    else:
        return img1[y_begin:y_begin+target_height, x_begin:x_begin+target_width], \
               img2[y_begin:y_begin+target_height, x_begin:x_begin+target_width], None

def resize(data, data_type, interpolation_method, target_height, target_width):

    if target_width == data.shape[1] and target_height == data.shape[0]:
       return data

    if data_type == 'stereo':
        if interpolation_method == 'bilinear':
            result =  util_cython.resize(data, target_width, target_height, 0.5)
        else:
            result = cv2.resize(data, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    else:
        result = np.zeros((target_height, target_width, 2))
        if interpolation_method == 'bilinear':
            result[:, :, 0] = util_cython.resize(data[:, :, 0], target_width, target_height, 0.5)
            result[:, :, 1] = util_cython.resize(data[:, :, 1], target_width, target_height, 0.5)
        else:
            result[:, :, 0] = cv2.resize(data[:, :, 0], (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            result[:, :, 1] = cv2.resize(data[:, :, 1], (target_width, target_height), interpolation=cv2.INTER_NEAREST)

    return result





