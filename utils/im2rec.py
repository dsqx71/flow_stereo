import os
import sys
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import numpy as np
import argparse
import cv2
import time
from utils.util import readPFM

def list_image(root, recursive, exts):
    image_list = []
    if recursive:
        cat = {}
        for path, subdirs, files in os.walk(root):
            print(len(cat), path)
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    image_list.append((len(image_list), os.path.relpath(fpath, root), cat[path]))
    else:
        for fname in os.listdir(root):
            fpath = os.path.join(root, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                image_list.append((len(image_list), os.path.relpath(fpath, root), 0))
    return image_list

def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i in xrange(len(image_list)):
            line = '%d\t'%image_list[i][0]
            for j in image_list[i][2:]:
                line += '%f\t'%j
            line += '%s\n'%image_list[i][1]
            fout.write(line)

def count_lines(path_out):
    with open(path_out,'r') as f:
        lens = len(f.readlines())
    return lens

def make_list(prefix_out, root, recursive, exts, num_chunks, train_ratio):
    image_list = list_image(root, recursive, exts)
    random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N+num_chunks-1)/num_chunks
    for i in xrange(num_chunks):
        chunk = image_list[i*chunk_size:(i+1)*chunk_size]
        if num_chunks > 1:
            str_chunk = '_%d'%i
        else:
            str_chunk = ''
        if train_ratio < 1:
            sep = int(chunk_size*train_ratio)
            write_list(prefix_out+str_chunk+'_train.lst', chunk[:sep])
            write_list(prefix_out+str_chunk+'_val.lst', chunk[sep:])
        else:
            write_list(prefix_out+str_chunk+'.lst', chunk)

def read_list(path_in):
    image_list = []
    with open(path_in) as fin:
        for line in fin.readlines():
            line = [i.strip() for i in line.strip().split('\t')]
            print line[0]
            item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            image_list.append(item)
    return image_list

def write_record(args, image_list):
    source = image_list
    tic = [time.time()]
    color_modes = {-1: cv2.IMREAD_UNCHANGED,
                    0: cv2.IMREAD_GRAYSCALE,
                    1: cv2.IMREAD_COLOR}
    total = len(source)

    def image_encode(item, q_out):

        # try:
        dirs = os.path.join(args.root, item[1])

        if '.pfm' in dirs:

            img, scale = readPFM(dirs)
            img = np.round(img * scale)

            tmp = np.zeros(img.shape + (3,))
            if args.type == 'stereo':
                # stereo
                img[img<0] = 0
                ori = img.copy()
                mask1 = img > 255
                mask2 = img > 510

                tmp[:, :, 2] = np.where(mask2, img - 510, 0)
                tmp[:, :, 1] = np.where(mask2, 255, np.where(mask1,img - 255, 0))
                tmp[:, :, 0] = np.where(mask1, 255, img)

                tmp[tmp>255] = 255
                tmp = np.round(tmp)
                tmp = tmp.astype(np.uint8)
                img = tmp
                # print  ori.max(), (img.sum(2)-ori).mean()
                    # plt.figure()
                    # plt.imshow(ori)
                    # plt.colorbar()
                    # plt.waitforbuttonpress()
                    # plt.figure()
                    # plt.imshow(img.sum(2))
                    # plt.colorbar()
                    # plt.waitforbuttonpress()
                    # plt.figure()
                    # plt.imshow((img.sum(2)-ori))
                    # plt.colorbar()
                    # plt.waitforbuttonpress()
            else:
                raise ValueError('don not support optical flow')
        else:
            img = cv2.imread(os.path.join(args.root, item[1]), color_modes[args.color])
        # except:
        #     print 'imread error:', item[1]
        #     return
        if img is None:
            print 'read none error:', item[1]
            return
        if args.center_crop:
            if img.shape[0] > img.shape[1]:
                margin = (img.shape[0] - img.shape[1])/2;
                img = img[margin:margin+img.shape[1], :]
            else:
                margin = (img.shape[1] - img.shape[0])/2;
                img = img[:, margin:margin+img.shape[0]]
        if args.resize:
            if img.shape[0] > img.shape[1]:
                newsize = (img.shape[0]*args.resize/img.shape[1], args.resize)
            else:
                newsize = (args.resize, img.shape[1]*args.resize/img.shape[0])
            img = cv2.resize(img, newsize)
        header = mx.recordio.IRHeader(0, 1, item[0], 0)

        try:
            s = mx.recordio.pack_img(header, img, quality=args.quality, img_fmt=args.encoding)
            q_out.put(('data', s, item))
        except:
            print 'pack_img error:',item[1]
            return

    def read_worker(q_in, q_out):
        while not q_in.empty():
            item = q_in.get()
            image_encode(item, q_out)

    def write_worker(q_out, prefix):
        pre_time = time.time()
        sink = []
        record = mx.recordio.MXRecordIO(prefix+'.rec', 'w')
        while True:
            stat, s, item = q_out.get()
            if stat == 'finish':
                write_list(prefix+'.lst', sink)
                break
            record.write(s)
            sink.append(item)
            if len(sink) % 1000 == 0:
                cur_time = time.time()
                print 'time:', cur_time - pre_time, ' count:', len(sink)
                pre_time = cur_time

    try:
        import multiprocessing

        q_in = [multiprocessing.Queue() for i in range(args.num_thread)]
        q_out = multiprocessing.Queue(1024)
        for i in range(len(image_list)):
            q_in[i % len(q_in)].put(image_list[i])
        read_process = [multiprocessing.Process(target=read_worker, args=(q_in[i], q_out)) \
                for i in range(args.num_thread)]

        for p in read_process:
            p.start()

        write_process = multiprocessing.Process(target=write_worker, args=(q_out,args.prefix))
        write_process.start()
        for p in read_process:
            p.join()
        q_out.put(('finish', '', []))
        write_process.join()
    except ImportError:
        print('multiprocessing not available, fall back to single threaded encoding')
        import Queue
        q_out = Queue.Queue()
        record = mx.recordio.MXRecordIO(args.prefix+'.rec', 'w')
        cnt = 0
        pre_time = time.time()
        for item in image_list:
            image_encode(item, q_out)
            if q_out.empty():
                continue
            _, s, _ = q_out.get()
            record.write(s)
            cnt += 1
            if cnt % 1000 == 0:
                cur_time = time.time()
                print 'time:', cur_time - pre_time, ' count:', cnt
                pre_time = cur_time

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Make an image record database by reading from\
        an image list or creating one')
    parser.add_argument('prefix', help='prefix of input/output files.')
    parser.add_argument('--root', default='',help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument('--list', type=bool, default=False,
        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument('--exts', type=list, default=['.jpeg','.jpg'],
        help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train_ratio', type=float, default=1.0,
        help='Ratio of images to use for training.')
    cgroup.add_argument('--recursive', type=bool, default=False,
        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')

    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument('--resize', type=int, default=0,
        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument('--center_crop', type=bool, default=False,
        help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument('--quality', type=int, default=80,
        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument('--num_thread', type=int, default=1,
        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument('--color', type=int, default=1, choices=[-1, 0, 1],
        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument('--encoding', type=str, default='.jpg', choices=['.jpg', '.png'],
        help='specify the encoding of the images.')

    rgroup.add_argument('--type', type=str, default='stereo', choices=['stereo', 'flow'],
                        help='type of label')

    args = parser.parse_args()

    image_list = read_list(args.prefix+'.lst')
    write_record(args, image_list)

if __name__ == '__main__':
    main()
