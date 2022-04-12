#!/usr/bin/env python

import os
import sys
import glob
import math
import uuid
import shutil
import pathlib
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import skimage
import torch
import tqdm
import gdal

import solaris as sol


def makeemptyfolder(path):
    """
    Create an empty folder, deleting anything there already
    """
    shutil.rmtree(path, ignore_errors=True)
    pathlib.Path(path).mkdir(exist_ok=True)


def readrotationfile(path):
    """
    Reads SAR_orientations file, which lists whether each strip was imaged
    from the north (denoted by 0) or from the south (denoted by 1).
    """
    rotationdf = pd.read_csv(path,
                             sep=' ',
                             index_col=0,
                             names=['strip', 'direction'],
                             header=None)
    rotationdf['direction'] = rotationdf['direction'].astype(int)
    return rotationdf


def lookuprotation(tilepath, rotationdf):
    """
    Looks up the SAR_orientations value for a tile based on its filename
    """
    tilename = os.path.splitext(os.path.basename(tilepath))[0]
    stripname = '_'.join(tilename.split('_')[-4:-2])
    rotation = rotationdf.loc[stripname].squeeze()
    return rotation


def copyrotateimage(srcpath, dstpath, rotate=False, deletesource=False):
    """
    Copying with rotation:  Copies a TIFF image from srcpath to dstpath,
    rotating the image by 180 degrees if specified.
    """
    #Handles special case where source path and destination path are the same
    if srcpath==dstpath:
        if not rotate:
            #Then there's nothing to do
            return
        else:
            #Move file to temporary location before continuing
            srcpath = srcpath + str(uuid.uuid4())
            shutil.move(dstpath, srcpath)
            deletesource = True

    if not rotate:
        shutil.copy(srcpath, dstpath, follow_symlinks=True)
    else:
        driver = gdal.GetDriverByName('GTiff')
        tilefile = gdal.Open(srcpath)
        copyfile = driver.CreateCopy(dstpath, tilefile, strict=0)
        numbands = copyfile.RasterCount
        for bandnum in range(1, numbands+1):
            banddata = tilefile.GetRasterBand(bandnum).ReadAsArray()
            banddata = np.fliplr(np.flipud(banddata)) #180 deg rotation
            copyfile.GetRasterBand(bandnum).WriteArray(banddata)
        copyfile.FlushCache()
        copyfile = None
        tilefile = None

    if deletesource:
        os.remove(srcpath)


def reorderbands(srcpath, dstpath, bandlist, deletesource=False):
    """
    Copies a TIFF image from srcpath to dstpath, reordering the bands.
    """
    #Handles special case where source path and destination path are the same
    if srcpath==dstpath:
        #Move file to temporary location before continuing
        srcpath = srcpath + str(uuid.uuid4())
        shutil.move(dstpath, srcpath)
        deletesource = True

    driver = gdal.GetDriverByName('GTiff')
    tilefile = gdal.Open(srcpath)
    geotransform = tilefile.GetGeoTransform()
    projection = tilefile.GetProjection()
    numbands = len(bandlist)
    shape = tilefile.GetRasterBand(1).ReadAsArray().shape
    copyfile = driver.Create(dstpath, shape[1], shape[0],
                             numbands, gdal.GDT_Byte)
    for bandnum in range(1, numbands+1):
        banddata = tilefile.GetRasterBand(bandlist[bandnum-1]).ReadAsArray()
        copyfile.GetRasterBand(bandnum).WriteArray(banddata)
    copyfile.SetGeoTransform(geotransform)
    copyfile.SetProjection(projection)
    copyfile.FlushCache()
    copyfile = None
    tilefile = None

    if deletesource:
        os.remove(srcpath)