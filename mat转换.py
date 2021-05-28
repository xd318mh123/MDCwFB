from osgeo import gdal, gdalconst
import os
import numpy as np
import scipy.io as scio
# def gdalRead( img_path):
#     ds = gdal.Open(img_path)
#     width = ds.RasterXSize
#     height = ds.RasterYSize
#     ds = ds.ReadAsArray(0, 0, width, height).astype(np.int16)
#     image_single = ds.reshape(1,1000,1000)
#     # image_single=image_single.transpose(1, 2, 0)
#     image = np.concatenate((image_single, image_single, image_single), axis=0)
#     print(image)
#     return image

def save_array_as_tif(array, fname,image_size,driver='GTiff', ):
    nbands = array.shape[0]
    driver = gdal.GetDriverByName(driver)
    tods = driver.Create(fname, image_size, image_size,nbands, gdal.GDT_UInt16,options=["INTERLEAVE=PIXEL"])
    # if array.ndim == 2:
    #     tods.GetRasterBand(1).WriteArray(array)
    # else:
    #     for i in range(nbands):
    #         tods.GetRasterBand(i + 1).WriteArray(array[i])
    band_list=[i for i in range(1,nbands+1)]
    tods.WriteRaster(0,0,image_size,image_size,array.tostring(),image_size,image_size,band_list=band_list)


if __name__ == "__main__":

    path = r"./result/MYNET.mat"
    data= scio.loadmat(path)
    data = scio.loadmat(path)['result']
    # data = scio.loadmat(path)["gt"]
    data=data *2047
    image=np.array(data).astype(np.int16)
    # image=np.expand_dims(image, 2)  #只针对一个波段
    # print(image.shape)
    # # # print(type(image))
    image=image.transpose(2,0,1)
    # # # ds = gdalRead(path)
    save_array_as_tif(image,"./MYNET.tif",image_size=256)


