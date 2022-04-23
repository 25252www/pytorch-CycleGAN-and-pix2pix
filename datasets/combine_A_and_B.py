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
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from gen_span2fused_dataset import spandir

def create_multiband_geotiff(array, out_name, proj, geo, nodata=0, out_format=gdal.GDT_Byte, verbose=False):
    """Convert an array to an output georegistered geotiff.
    Arguments
    ---------
    array : numpy array
        A numpy array with a the shape: [Channels, X, Y] or [X, Y]
    out_name : str
        The output name and path for your image
    proj : gdal projection
        A projection, can be extracted from an image opened with gdal with image.GetProjection().  Can be set to None if no georeferencing is required.
    geo : gdal geotransform
        A gdal geotransform which indicates the position of the image on the earth in projection units. Can be set to None if no georeferencing is required.
        Can be extracted from an image opened with gdal with image.GetGeoTransform()
    nodata : int, default - 0
        A value to set transparent for GIS systems. Can be set to None if the nodata value is not required.
    out_format : gdalconst
        https://gdal.org/python/osgeo.gdalconst-module.html
        Must be one of the variables listed in the docs above
    verbose : bool
        A verbose output, printing all inputs and outputs to the function.  Useful for debugging.
    """
    driver = gdal.GetDriverByName('GTiff')
    # 如果shape为(900,900),则转换为(1,900,900)
    if len(array.shape) == 2:
        array = array[np.newaxis, ...]
        # print("np.shape==2")
        # print("shape after shape ", np.shape(array))
        # print("array after shape",array)
    os.makedirs(os.path.dirname(os.path.abspath(out_name)), exist_ok=True)
    # 这步只是创建图像（存储空间已经被分配到硬盘上了），之后才往图像里写数据
    # GDALDataType的默认类型为GDT_Byte
    dataset = driver.Create(
        out_name, array.shape[2], array.shape[1], array.shape[0], out_format)
    if verbose is True:
        print("Array Shape, should be [Channels, X, Y] or [X,Y]:", array.shape)
        print("Output Name:", out_name)
        print("Projection:", proj)
        print("GeoTransform:", geo)
        print("NoData Value:", nodata)
        print("Bit Depth:", out_format)
    if proj is not None:
        dataset.SetProjection(proj)
    if geo is not None:
        dataset.SetGeoTransform(geo)
    if nodata is None:
        for i, image in enumerate(array, 1):
            dataset.GetRasterBand(i).WriteArray(image)
        del dataset
    else:
        for i, image in enumerate(array, 1):  # 下标从 1 开始
            # print("i",i)
            # print(type(image))
            dataset.GetRasterBand(i).WriteArray(image)
            dataset.GetRasterBand(i).SetNoDataValue(nodata)
        del dataset

# 从A、B文件夹生成AB文件夹


def image_write(path_A, path_B, path_AB, spanmin, spanmax):
    tilefile = gdal.Open(path_A)
    proj = tilefile.GetProjection()
    geo = tilefile.GetGeoTransform()
    band = tilefile.GetRasterBand(1)
    im_A = band.ReadAsArray().astype(np.float32)
    im_B = cv2.imread(path_B, 1)
    # 从BGR通道转成RGB，便于之后保存成tiff格式
    im_B = cv2.cvtColor(im_B, cv2.COLOR_BGR2RGB)
    # Min-Max Scaling to [0,255]
    im_A = (im_A - spanmin)*255 / (spanmax - spanmin)
    im_A = im_A.astype(np.float32)
    # im_A(900,900)->(900,900,3)
    im_A = np.expand_dims(im_A, axis=2).repeat(3, axis=2)
    # 保存成tiff
    # im_AB(900,1800,3)->(3,900,1800)
    im_AB = np.concatenate([im_A, im_B], 1)
    im_AB = np.swapaxes(im_AB, 1, 0)
    im_AB = np.swapaxes(im_AB, 2, 0)
    create_multiband_geotiff(im_AB, path_AB, proj, geo,
                             nodata=0, out_format=gdal.GDT_Float32)


parser = argparse.ArgumentParser('create image pairs')
parser.add_argument('--fold_A', dest='fold_A', help='input directory for image A',
                    type=str, default='../dataset/50kshoes_edges')
parser.add_argument('--fold_B', dest='fold_B', help='input directory for image B',
                    type=str, default='../dataset/50kshoes_jpg')
parser.add_argument('--fold_AB', dest='fold_AB',
                    help='output directory', type=str, default='../dataset/test_AB')
parser.add_argument('--num_imgs', dest='num_imgs',
                    help='number of images', type=int, default=1000000)
parser.add_argument('--use_AB', dest='use_AB',
                    help='if true: (0001_A, 0001_B) to (0001_AB)', action='store_true')
parser.add_argument('--no_multiprocessing', dest='no_multiprocessing',
                    help='If used, chooses single CPU execution instead of parallel execution', action='store_true', default=False)
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
    pool = Pool()

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
    for n in tqdm.tqdm(range(num_imgs)):
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
                pool.apply_async(image_write, args=(
                    path_A, path_B, path_AB, spanmin, spanmax))
            else:
                image_write(path_A, path_B, path_AB, spanmin, spanmax)
        # break;
if not args.no_multiprocessing:
    pool.close()
    pool.join()
