import os
import pickle
import cv2
import torch
import torchvision.transforms as transforms
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np


def construct_train_txt(address):
    file_name_lst = os.listdir(address)
    with open("../data/train.txt", "w+") as txt:
        for file_name in file_name_lst:
            line = "../data/train/" + file_name + "\n"
            txt.writelines(line)


def construct_val_txt(address):
    file_name_lst = os.listdir(address)
    with open("../data/val.txt", "w+") as txt:
        for file_name in file_name_lst:
            line = "../data/valid/" + file_name + "\n"
            txt.writelines(line)


def get_minimal_grid(address):
    file_lst = os.listdir(address)
    minimize_pixel = np.inf
    for i, file in enumerate(file_lst):
        print("i = ", i)
        img = cv2.imread(address + file, cv2.IMREAD_COLOR)
        pixel_num = img.shape[0] * img.shape[1]
        if pixel_num < minimize_pixel:
            minimize_pixel = pixel_num
    print(minimize_pixel)


def cal_gaussian_kernel(sigma, size):
    """
    calculate gaussian blur kernel according to opencv
    https://blog.csdn.net/wuqindeyunque/article/details/103694900
    """
    kernel = np.zeros([size, size], dtype=float)
    if sigma <= 0:
        sigma = 0.3 * ((size - 1) / 2 - 1) + 0.8
    center = (size - 1) / 2
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-((i - center)**2 + (j - center) ** 2) / 2 / sigma ** 2)
    kernel /= np.sum(kernel)
    return kernel


def pre_process(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    print("origin shape = ", img.shape)

    # ##################################################################
    # crop 200 x 200 pixel
    # #################################################################
    anchor_width = np.random.randint(0, img.shape[0] - 200)
    anchor_height = np.random.randint(0, img.shape[1] - 200)
    img = img[anchor_width:anchor_width + 200, anchor_height:anchor_height + 200, :]

    # ################################################################
    # gaussian matrix size, width(sigma)
    # ################################################################
    blur_img = cv2.GaussianBlur(img, (15, 15), sigmaX=0.5, sigmaY=0.5)
    origin_row = blur_img.shape[0]
    origin_col = blur_img.shape[1]

    # ##################################################################
    # downsamplint
    # ##################################################################
    down_scale = 4
    blur_img = cv2.resize(blur_img, (origin_col // down_scale, origin_row // down_scale), interpolation=cv2.INTER_CUBIC)
    blur_img = cv2.resize(blur_img, (origin_col, origin_row), interpolation=cv2.INTER_CUBIC)
    print(cv2.getGaussianKernel(3, 80))
    print(cal_gaussian_kernel(80, 3))
    blur_img = blur_img.astype(float)
    blur_img = np.array(blur_img, dtype=float) / 255.0
    # ############################################################
    # add gaussian noise
    # ##############################################################
    noise = np.random.randn(*img.shape) / 255.0
    blur_img += noise
    if blur_img.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(blur_img, low_clip, 1.0)
    blur_img = np.uint8(out * 255)
    show_img = np.hstack((img, blur_img))
    cv2.namedWindow("blur_img", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("src", cv2.WINDOW_NORMAL)
    # cv2.imshow("src", img)
    cv2.imshow("blur_img", show_img)
    cv2.waitKey(0)
    return img

def frequent_show(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    freq = np.fft.fft2(img, axes=(0, 1))
    freq = np.fft.fftshift(freq)
    cv2.namedWindow("frequent")
    cv2.imshow("frequent", freq)
    cv2.waitKey(0)


def traversal_total_dir(input_path, output_path):
    for root, dirs, files in os.walk(input_path):
        for name in files:
            address = os.path.join(root, name)
            print("address = ", address)
            out_img = pre_process(address)
            cv2.imwrite(output_path + name, out_img)


def plot_precision_recall_curve(address):
    with open(address, "rb+") as f:
        dis_label_lst = pickle.load(f)
    print(len(dis_label_lst))
    dis_total = dis_label_lst[0][0]
    label_total = dis_label_lst[0][0]

    for i, dis_label in enumerate(dis_label_lst):
        if i == 0:
            continue
        dis = dis_label[0]
        label = dis_label[1]
        dis_total = torch.cat([dis_total, dis], dim=0)
        label_total = torch.cat([label_total, label], dim=0)
    label_total = label_total.numpy().astype(int)
    dis_total = dis_total.numpy()
    max_dis = np.max(dis_total)
    # dis_total = 1 - dis_total / max_dis
    precision, recall, thresh = precision_recall_curve(label_total, dis_total, pos_label=1)
    print(precision)
    print(recall)
    print(thresh)
    # print((1 - thresh) * max_dis)
    plt.figure(1)  # ????????????1
    plt.title('Precision/Recall Curve')  # give plot a title
    plt.xlabel('Recall')  # make axis labels
    plt.ylabel('Precision')
    plt.plot(recall, precision)
    plt.show()


if __name__ == "__main__":
    # construct_train_txt("../data/train/")
    # construct_val_txt("../data/valid/")
    # get_minimal_grid("../data/train/")
    pre_process("../data/train/0001.png")
    # frequent_show("../data/train/0001.png")


    # dump_files("../../data/process", "../../data/")
    # load_pkl("../../data")
    # construct_pos_neg_samples_test()
    # plot_precision_recall_curve("dis_label.pkl")