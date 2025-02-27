import glob
import os
import numpy as np
import torch
from utils.images_utils import imagesc, to_8bit
import torchvision.transforms as transforms
from PIL import Image
import scipy.ndimage
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
import scipy.ndimage
from skimage import measure
from utils.match_dess_tse import linear_registration, make_compare, apply_warp
import shutil
from models.unet import UNet_clean
from models.OaiLocator import OaiLocator
from dotenv import load_dotenv
import tifffile as tiff


def seg2crop(npys, cropHW, cartilage_channels):
    """
    Using the segmented shape of cartilage to create the cropping coordinates
    :param npys: Numpy array containing segmented cartilage shape
    :param cropHW: List [cropH, cropW] specifying the height and width of the crop window
    :param cartilage_channels: List of cartilage channel values in npys
    :return: List [crop_start_row, crop_end_row, crop_start_column, crop_end_column] specifying the cropping coordinates
    """
    # Generate front view of the segmented cartilage shape
    front = (sum([(npys == c) for c in cartilage_channels])).sum(2)
    front[:20, :] = 0
    front[-20:, :] = 0
    front[:, :20] = 0
    front[:, -20:] = 0

    # Calculate moving average of the front view
    movingavg = scipy.ndimage.uniform_filter(front / front.max(), size=50)

    # Find the coordinates of the maximum value in the moving average
    (cm0front, cm1front) = np.unravel_index(movingavg.argmax(), movingavg.shape)

    # Calculate cropping coordinates based on the maximum value coordinates
    crop = [cm0front + cropHW[0], cm0front + cropHW[1], cm1front + cropHW[2], cm1front + cropHW[3]]

    # adjust cropping coordinate if the cropping window is out of the boundaries
    (H, W, _) = npys.shape
    if crop[0] < 0:
        crop[0] = 0
        crop[1] = (cropHW[1] - cropHW[0])
    elif crop[1] > H:
        crop[1] = H
        crop[0] = H - (cropHW[1] - cropHW[0])
    if crop[2] < 0:
        crop[2] = 0
        crop[3] = (cropHW[3] - cropHW[2])
    elif crop[3] > W:
        crop[3] = W
        crop[2] = W - (cropHW[3] - cropHW[2])
    return crop


def get_segmentation(x0, model_seg, model_t2d=None):
    """
        Get the segmentation mask for input image x0 using the segmentation model
        :param x0: Input image as a numpy array
        :param model_seg: Segmentation model
        :param model_t2d: Optional model for additional processing
        :return: Segmentation mask as a numpy array
        """
    # Convert input image to tensor
    x = 1 * x0
    x = npy_2_tensor(x)
    x = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(x)

    # Crop the input image into (384,384)
    dx = (x.shape[2] - 384) // 2
    x_cropped = x[:, :, dx:-dx, dx:-dx]

    # Apply additional processing if model_t2d is provided
    if model_t2d is not None:
        x_cropped = model_t2d(x_cropped.cuda())

    # Generate the segmentation mask using the segmentation model
    seg_cropped = model_seg(x_cropped)
    seg_cropped = torch.argmax(seg_cropped, 1).detach().cpu()

    # Create a segmentation mask with the same size as the input image
    seg = torch.zeros((x.shape[0], x.shape[2], x.shape[3]))
    seg[:, dx:-dx, dx:-dx] = seg_cropped
    seg = seg.permute(1, 2, 0).numpy()

    return seg


def read_slices_to_3d(folder):
    """
    Read a sequence of 2D slices from a folder and concatenate them to create a 3D volume
    :param folder: Folder containing the 2D slices
    :return: 3D volume as a numpy array
    """
    all = []
    slices_list = sorted(glob.glob(folder + '/*'))
    for i in range(len(slices_list)):
        x = np.load(slices_list[i])
        all.append(np.expand_dims(x, 2))
    return np.concatenate(all, 2)


def npy_2_tensor(y0, threshold=None):
    """
    Convert a numpy array to a PyTorch tensor
    :param y0: Input numpy array
    :param threshold: Optional threshold value to limit the array values
    :return: PyTorch tensor
    """
    y = 1 * y0
    if threshold:
        y[y >= threshold] = threshold
    y = y / y.max()
    y = torch.from_numpy(y)
    y = y.permute(2, 0, 1)
    y = y.unsqueeze(1)
    y = torch.cat([y] * 3, 1)
    y = y.type(torch.FloatTensor)
    return y


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def get_pcl(npys, model_pcl, threshold=None):
    """
    get slice location of PCL
    """
    y = npy_2_tensor(npys, threshold=threshold)
    loc, = model_pcl((y,))
    loc = nn.Softmax()(loc).detach().cpu().numpy()
    pcl = np.argmax(loc[10:-10, 1]) + 9  # drop the first and last 10 slices
    return pcl


def get_model():
    model_seg = torch.load(os.environ.get('model_seg'))
    model_t2d = torch.load(os.environ.get('model_t2d'))
    model_pcl = torch.load(os.environ.get('model_pcl'))
    model_t2d.eval()
    model_seg.eval()
    return model_seg, model_pcl, model_t2d


def get_seg(y0, unet, resample, t2d=None):
    osize = y0.shape[2]
    y = 1 * y0
    y = torch.nn.functional.interpolate(y, (resample, resample), mode='bicubic', align_corners=True)

    for i in range(y.shape[0]):
        y[i, :, :, :] = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(y[i, :, :, :])

    y = y.cuda()
    y = unet(y)
    y = torch.nn.functional.interpolate(y, (osize, osize), mode='bicubic', align_corners=True).detach().cpu().numpy()
    seg = np.argmax(y, 1)
    return seg


class OAI_preprocess():
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.mri = dict()

    def get_mri(self, sequence, subject, size, threshold=None):
        mri = read_slices_to_3d(folder=os.path.join(self.source, sequence, subject))
        if mri.shape[:2] != size:
            mri = scipy.ndimage.zoom(mri, (size[0] / mri.shape[0], size[1] / mri.shape[1], 1))
        if sequence.split('_')[-1] == 'RIGHT':
            mri = mri[:, :, ::-1]
        if threshold is not None:
            mri[mri >= threshold] = threshold
        self.mri[sequence] = mri

    def crop_npys(self, list_names, list_crop):
        """
        crop the list of npys using the same coordinates together
        """
        for name in list_names:
            npy = self.mri[name]
            if list_crop[0] is not None:
                npy = npy[list_crop[0], :, :]
            if list_crop[1] is not None:
                npy = npy[:, list_crop[1], :]
            if list_crop[2] is not None:
                npy = npy[:, :, list_crop[2]]
            self.mri[name] = npy


if __name__ == '__main__':
    load_dotenv('.env')
    from utils.oai_zip_utils import meta_process

    # get pytorch models
    # model_seg, model_pcl, model_t2d = get_model()
    # basic info
    source = '/media/ExtHDD02/OAIDataBase/womac4min0/'
    destination = source + 'raw3D/'
    for sequence in ['SAG_IW_TSE', 'COR_IW_TSE']:
        for side in ['_LEFT', '_RIGHT']:
            seq_side = sequence + side
            print(seq_side)
            op = OAI_preprocess(source=source + 'dcm/Npy/', destination=destination)
            # subjects_list = [x.split('/')[-1] for x in sorted(glob.glob(source + 'dcm/Npy/' + seq_side + '/*'))]
            # get id from glob & remove duplicate

            subjects_list = [i.split('/')[-1] for i in sorted(glob.glob(source + 'dcm/Npy/' + seq_side + '/*'))]
            print('subjects_list', subjects_list[:5])
            # subjects_list = list(set(subjects_list))
            # create destination
            os.makedirs(destination, exist_ok=True)
            os.makedirs(os.path.join(destination, sequence), exist_ok=True)
            os.makedirs(os.path.join(destination, 'check'), exist_ok=True)

            for subject in subjects_list[:]:
                # get op.tse
                op.get_mri(sequence=seq_side, subject=subject, size=(444, 444))
                # op.pcl_tse = get_pcl(op.mri[sequence], model_pcl, threshold=None)

                # select slices by the location of pcl
                # op.crop_npys(list_names=[sequence],
                #              list_crop=[None, None, range(op.pcl_tse - 11, op.pcl_tse + 12)])12

                # get bone and cartilage segmentation
                # op.mri[sequence + '_seg'] = get_segmentation(op.mri[sequence], model_seg=model_seg, model_t2d=model_t2d)

                # crop by cartilage segmentation
                # dx = 384 // 2
                # crop = seg2crop(npys=op.mri[sequence + '_seg'], cropHW=[-dx - 100, dx - 100, -dx - 30, dx - 30],
                #                 cartilage_channels=[2, 4])
                # op.crop_npys(list_names=[sequence, sequence + '_seg'],
                #              list_crop=[range(crop[0], crop[1]), range(crop[2], crop[3]), None])

                # save as npy
                #np.save(os.path.join(destination, 'TSE', subject + '_' + sequence.split('_')[-1] + '.npy'), op.mri[sequence])`

                # save as tif
                npy = np.transpose(op.mri[seq_side], (2, 0, 1))
                tiff.imsave(os.path.join(destination, sequence, subject + side + '.tif'), npy)

        # print check image
        # front = op.mri[sequence+'_seg'].sum(2)
        # pcl = op.mri[sequence][:, :, 11]
        # check = np.concatenate([x / x.max() for x in [front, pcl]], 1)
        # imagesc(check, show=False,
        #         save=os.path.join(destination, 'check', subject + '_' + sequence.split('_')[-1] + '.png'))