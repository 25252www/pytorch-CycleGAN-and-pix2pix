import os
import glob
from utils import *


datasetdir = "/home/liuxiangyu/dataset/spacenet6/train/AOI_11_Rotterdam"
rotationfile = datasetdir+"/SummaryData/SAR_orientations.txt"

spandir = datasetdir + "/SPAN"
fuseddir = datasetdir + "/Fused-RGB-SAR"
sardir = datasetdir + "/SAR-Intensity"

localdatasetdir = "/home/liuxiangyu/pytorch-CycleGAN-and-pix2pix/datasets/span2fused"
localdatasetdirAtrain = localdatasetdir+"/A/train"
localdatasetdirAval = localdatasetdir+"/A/val"
localdatasetdirAtest = localdatasetdir+"/A/test"
localdatasetdirBtrain = localdatasetdir+"/B/train"
localdatasetdirBval = localdatasetdir+"/B/val"
localdatasetdirBtest = localdatasetdir+"/B/test"


# todo: 把SPAN归一化到[0,255]，否则Fused-RGB-SAR一片黑(趋近于0)
def gen_dataset():
    # 创建文件夹
    folders = [localdatasetdirAtrain,
               localdatasetdirAval,
               localdatasetdirAtest,
               localdatasetdirBtrain,
               localdatasetdirBval,
               localdatasetdirBtest]
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
    # 原span文件名
    spanpaths = glob.glob(os.path.join(spandir,'*.tif'))
    spanbasenames = [os.path.basename(i) for i in spanpaths]
    spanbasenames.sort() 
    # 读取SAR_orientations.txt
    assert(rotationfile is not None)
    rotationdf = readrotationfile(rotationfile)
    # 
    ledge = 591550 #Approximate west edge of training data area
    redge = 596250 #Approximate east edge of training data area
    numgroups=5
    validationgroup = numgroups - 1
    # 遍历全部源数据，旋转并复制到各个文件夹下
    for i, spanbasename in enumerate(tqdm.tqdm(spanbasenames)):
        rotationflag = lookuprotation(spanbasename, rotationdf)
        rotationflagbool = rotationflag == 1
        # 生成文件名
        localbasename = spanbasename.split("SPAN")[0] + "span2fused" + spanbasename.split("SPAN")[1]
        # 生成A/test全部文件
        Asrc = os.path.join(spandir, spanbasename)
        Atestdst =os.path.join(localdatasetdirAtest, localbasename)
        copyrotateimage(Asrc, Atestdst, rotate=rotationflagbool)
        # 生成B/test全部文件
        fusedbasename = spanbasename.split("SPAN")[0] + "SAR-Intensity-Colorized-HSV" + spanbasename.split("SPAN")[1]
        Bsrc = os.path.join(fuseddir, fusedbasename)
        Btestdst = os.path.join(localdatasetdirBtest, localbasename)
        copyrotateimage(Bsrc, Btestdst, rotate=rotationflagbool)
        #数据集划分
        sarbasename = spanbasename.split("SPAN")[0] + "SAR-Intensity" + spanbasename.split("SPAN")[1]
        sarpath = os.path.join(sardir,sarbasename)
        sarfile = gdal.Open(sarpath)
        sartransform = sarfile.GetGeoTransform()
        sarx = sartransform[0]
        groupnum = min(numgroups-1, max(0, math.floor((sarx-ledge) / (redge-ledge) * numgroups)))
        # 放入A/val & B/val
        if groupnum == validationgroup:
            Avaldst = os.path.join(localdatasetdirAval,localbasename)
            copyrotateimage(Asrc, Avaldst, rotate=rotationflagbool)
            Bvaldst = os.path.join(localdatasetdirBval,localbasename)
            copyrotateimage(Bsrc, Bvaldst, rotate=rotationflagbool)
        # 放入A/train & B/train
        else:
            Atraindst = os.path.join(localdatasetdirAtrain,localbasename)
            copyrotateimage(Asrc, Atraindst, rotate=rotationflagbool)
            Btraindst = os.path.join(localdatasetdirBtrain,localbasename)
            copyrotateimage(Bsrc, Btraindst, rotate=rotationflagbool)
    

    
if __name__ == "__main__":
    gen_dataset()
    """
    num of
    A train & B train: 2696
    A val & B val: 705
    A test & B test: 3401
    """