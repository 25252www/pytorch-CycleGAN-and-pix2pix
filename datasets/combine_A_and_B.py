from hashlib import sha1
import os
import numpy as np
import cv2
import argparse
import glob
import gdal
import tqdm
from multiprocessing import Pool
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)
from gen_span2fused_dataset import spandir

# 从A、B文件夹生成AB文件夹
def image_write(path_A, path_B, path_AB, spanmin, spanmax):
    tilefile = gdal.Open(path_A)
    band = tilefile.GetRasterBand(1)
    im_A = band.ReadAsArray().astype(np.float32)
    im_B = cv2.imread(path_B, 1) # python2: cv2.CV_LOAD_IMAGE_COLOR; python3: cv2.IMREAD_COLOR
    # Min-Max Scaling to [0,255]
    im_A = (im_A - spanmin)*255 / (spanmax - spanmin)
    im_A = im_A.astype(np.float32)
    # im_A(900,900)->(900,900,3)
    im_A = np.expand_dims(im_A,axis=2).repeat(3,axis=2)
    im_AB = np.concatenate([im_A, im_B], 1)
    # print(np.shape(im_AB))
    # print(im_AB.dtype)
    # print(im_AB)
    cv2.imwrite(path_AB, im_AB)

parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A', type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B', type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB', help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs', help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB', help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing', help='If used, chooses single CPU execution instead of parallel execution', action='store_true',default=False)
args = parser.parse_args()

for arg in vars(args):
    print('[%s] = ' % arg, getattr(args, arg))


# # find the min and max of the whole span dataset.
# spanpaths = glob.glob(os.path.join(spandir,'*.tif'))
# tilefile = gdal.Open(spanpaths[0])
# band = tilefile.GetRasterBand(1)
# spanmin = band.ComputeRasterMinMax()[0]
# spanmax = band.ComputeRasterMinMax()[1]
# for i, spanpath in enumerate(tqdm.tqdm(spanpaths)):
#     tilefile = gdal.Open(spanpath)
#     band = tilefile.GetRasterBand(1)
#     minandmax = band.ComputeRasterMinMax()
#     spanmin = min(spanmin,minandmax[0])
#     spanmax = max(spanmax,minandmax[1])
# print("min of the whole span dataset",spanmin)
# print("max of the whole span dataset",spanmax)
spanmin = 0
spanmax = 16984.333984375

splits = os.listdir(args.fold_A)

if not args.no_multiprocessing:
    pool=Pool()

for sp in splits:
    img_fold_A = os.path.join(args.fold_A, sp)
    img_fold_B = os.path.join(args.fold_B, sp)
    img_list = os.listdir(img_fold_A)
    if args.use_AB:
        img_list = [img_path for img_path in img_list if '_A.' in img_path]

    num_imgs = min(args.num_imgs, len(img_list))
    print('split = %s, use %d/%d images' % (sp, num_imgs, len(img_list)))
    img_fold_AB = os.path.join(args.fold_AB, sp)
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    print('split = %s, number of images = %d' % (sp, num_imgs))
    for n in range(num_imgs):
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        if args.use_AB:
            name_B = name_A.replace('_A.', '_B.')
        else:
            name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            name_AB = name_A
            if args.use_AB:
                name_AB = name_AB.replace('_A.', '.')  # remove _A
            path_AB = os.path.join(img_fold_AB, name_AB)
            if not args.no_multiprocessing:
                pool.apply_async(image_write, args=(path_A, path_B, path_AB, spanmin, spanmax))
            else:
                image_write(path_A, path_B, path_AB, spanmin, spanmax)
        # break;
if not args.no_multiprocessing:
    pool.close()
    pool.join()
