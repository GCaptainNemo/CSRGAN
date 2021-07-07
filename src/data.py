import torch.utils.data.dataset as dataset
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset
import torch.fft as fft


class DataSampler(Dataset):
    def __init__(self, txt, batch_size):
        super(DataSampler, self).__init__()
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
        origin_row = blur_img.shape[0]
        origin_col = blur_img.shape[1]

        # ###############################################
        # down scale
        # ################################################

        scale_factor = np.random.random(1) * 3 + 1
        blur_img = cv2.resize(blur_img, (origin_col // scale_factor, origin_row // scale_factor),
                              interpolation=cv2.INTER_CUBIC)
        blur_img = np.array(blur_img, dtype=float) / 255

        # ###############################################
        # add noise
        # ################################################

        noise_sigma = np.random.random(1)
        noise = np.random.normal(0, noise_sigma ** 0.5, blur_img.shape)
        blur_img += noise
        if blur_img.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(blur_img, low_clip, 1.0)
        blur_img = np.uint8(out * 255)

        # #############################################
        # back size
        # #############################################

        lr = cv2.resize(blur_img, (origin_col, origin_row), interpolation=cv2.INTER_CUBIC)
        return lr, blur_sigma, scale_factor, noise_sigma

    def __getitem__(self, item):
        address = self.content[item].strip()

        # ###########################################
        # randomly choose 200x200 pixel image
        # ###########################################

        cv_src_img = cv2.imread(address, cv2.IMREAD_COLOR)
        anchor_width = np.random.randint(0, cv_src_img.shape[0] - 200)
        anchor_height = np.random.randint(0, cv_src_img.shape[1] - 200)
        cv_src_img = cv_src_img[anchor_width:anchor_width + 200, anchor_height:anchor_height + 200, :]

        lr, blur_sigma, scale_factor, noise_sigma = self.get_lr_img(cv_src_img)
        hr_img = transforms.ToTensor()(cv_src_img)
        lr_img = transforms.ToTensor()(lr)
        # #########################################################################
        # extend to physical parameters tensor
        # ########################################################################
        blur_tensor = torch.unsqueeze(torch.ones(hr_img.shape[1:], dtype=torch.float32) * blur_sigma, dim=0)
        scale_tensor = torch.unsqueeze(torch.ones(hr_img.shape[1:], dtype=torch.float32) * scale_factor, dim=0)
        noise_tensor = torch.unsqueeze(torch.ones(hr_img.shape[1:], dtype=torch.float32) * noise_sigma, dim=0)

        # discriminator input (HR, BLUR, SCALE, NOISE, IR)
        true_hr_phy_lr = torch.cat([hr_img, blur_tensor, scale_tensor, noise_tensor, lr_img], dim=0).to(torch.float32)

        # generator input (lr_img)
        generator_input = lr_img
        return true_hr_phy_lr, generator_input

