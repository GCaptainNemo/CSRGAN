import torch.utils.data.dataset as dataset
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset


class DataSamplerTrain(Dataset):
    def __init__(self, txt, batch_size):
        super(DataSamplerTrain, self).__init__()
        self.load_txt(txt)
        self.batch_size = batch_size

    def load_txt(self, txt_address):
        with open(txt_address, "r+") as f:
            self.content = f.readlines()

    def __len__(self):
        return len(self.content)

    def get_lr_img(self, hr_img):
        # [0 - 50] uniform distribution
        # #######################################################
        # gaussian filter
        # ########################################################

        blur_sigma = float(np.random.random(1) * 50)
        blur_img = cv2.GaussianBlur(hr_img, (15, 15), blur_sigma)
        # origin_row = blur_img.shape[0]
        # origin_col = blur_img.shape[1]

        # ###############################################
        # down scale
        # ################################################

        # scale_factor = np.random.random(1) * 3 + 1
        # blur_img = cv2.resize(blur_img, (int(origin_col // scale_factor), int(origin_row // scale_factor)),
        #                       interpolation=cv2.INTER_CUBIC)
        blur_img = np.array(blur_img, dtype=float) / 255

        # ###############################################
        # add noise
        # ################################################

        # noise_sigma = np.random.random(1)
        # noise = np.random.normal(0, noise_sigma ** 0.5, blur_img.shape)
        # blur_img += noise
        if blur_img.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(blur_img, low_clip, 1.0)
        blur_img = np.uint8(out * 255)

        # #############################################
        # back size
        # #############################################

        # lr = cv2.resize(blur_img, (origin_col, origin_row), interpolation=cv2.INTER_CUBIC)
        # return lr, blur_sigma, scale_factor, noise_sigma
        return blur_img, blur_sigma


    def __getitem__(self, item):
        address = self.content[item].strip()

        # ###########################################
        # randomly choose 572x572 pixel image
        # ###########################################

        cv_src_img = cv2.imread(address, cv2.IMREAD_COLOR)
        anchor_width = np.random.randint(0, cv_src_img.shape[0] - 512)
        anchor_height = np.random.randint(0, cv_src_img.shape[1] - 512)
        cv_src_img = cv_src_img[anchor_width:anchor_width + 512, anchor_height:anchor_height + 512, :]

        # lr, blur_sigma, scale_factor, noise_sigma = self.get_lr_img(cv_src_img)
        lr, blur_sigma = self.get_lr_img(cv_src_img)

        true_hr = transforms.ToTensor()(cv_src_img)
        lr_img = transforms.ToTensor()(lr)
        # physical_tensor = torch.tensor([blur_sigma, scale_factor, noise_sigma], dtype=torch.float32)
        physical_tensor = torch.tensor([blur_sigma], dtype=torch.float32)

        # #########################################################################
        # extend to physical parameters tensor
        # ########################################################################

        # discriminator input (HR, BLUR, SCALE, NOISE, IR)

        return true_hr, physical_tensor, lr_img

