import torch
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torchvision import transforms
from src.data import DataSamplerTrain
from torch.utils.data import DataLoader
from skimage.measure import compare_psnr, compare_ssim, compare_mse



class InferenceDataset(Dataset):
    def __init__(self, inference_txt_address):
        super(InferenceDataset, self).__init__()
        self.load_txt(inference_txt_address)

    def load_txt(self, inference_txt_address):
        with open(inference_txt_address, "r+") as f:
            self.content = f.readlines()

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        content = self.content[item].strip()
        lst = content.split("    ")
        address1 = lst[0]
        # print("address1 = ", address1)
        img1 = transforms.ToTensor()(cv2.imread(address1, cv2.IMREAD_GRAYSCALE))
        address2 = lst[1]
        img2 = transforms.ToTensor()(cv2.imread(address2, cv2.IMREAD_GRAYSCALE))
        label = torch.tensor([int(lst[2])], dtype=torch.float32)
        print("label = ", lst[2])
        return img1, img2, label


def confusion_matrix(preds, labels, conf_matrix):
    # confusion matrix
    # yaxis - gt; xaxis - pred
    for gt, pred in zip(labels, preds):
        conf_matrix[int(round(gt.item())), int(round(pred.item()))] += 1
    return conf_matrix


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : confusion matrix
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:percentage, False:Num
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")
    plt.tight_layout()
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    plt.show()

def plot_wrong_img_pairs(address):
    with open(address, "rb+") as f:
        wrong_lst = pickle.load(f)
    print(len(wrong_lst))
    print(type(wrong_lst[0][0]))
    pil = transforms.ToPILImage()
    for i, img_pair in enumerate(wrong_lst):
        if i % 8 == 0:
            imgs = np.vstack([img_pair[0], img_pair[1]])
        else:
            img = np.vstack([img_pair[0], img_pair[1]])
            imgs = np.hstack([imgs, img])
        if i % 8 == 7:
            # imgs = (imgs * 255).astype(int)
            print(imgs)
            cv2.imshow("img", imgs)
            # imgs = pil(imgs)
            cv2.waitKey(0)
            # if imgs.mode == "F":
            #     imgs = imgs.convert('L')
            # imgs.save("../../result/{}.jpg".format(i // 8))
            cv2.imwrite("../../result/{}.jpg".format(i // 8), imgs * 255)


if __name__ == "__main__":
    txt_address = "../data/train.txt"
    data_loader = DataLoader(DataSamplerTrain(txt=txt_address, batch_size=1), batch_size=1)
    device = torch.device("cuda:0")
    # generator_nn = torch.load("../result/model-v2/generator-2899.pth", map_location=device)
    generator_nn = torch.load("../result/model-v3/generator-4499.pth", map_location=device)

    # generator_nn = torch.load("generator-399.pth", map_location=device)
    # generator_nn.to(device)
    # generator_nn.to("cpu")
    generator_nn = generator_nn.eval()
    # generator_nn.extention.to("cpu")
    total_psnr = 0
    total_ssim = 0
    psnr_lst = []
    ssim_lst = []
    mse_physical_par_lst = []
    for i, (hr, phy_par, lr) in enumerate(data_loader):
        sr, inf_phy_par = generator_nn(lr.cuda())
        sr = torch.squeeze(sr, dim=0).detach()
        inf_phy_par = inf_phy_par.detach()
        print(torch.max(lr))
        print("img.shape = ", sr.shape)
        pil_img = transforms.ToPILImage()(sr)
        print(pil_img)
        print(type(pil_img))
        # img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
        sr = np.asarray(pil_img)

        lr = torch.squeeze(lr, dim=0)
        pil_img_lr = transforms.ToPILImage()(lr)
        # img_lr = cv2.cvtColor(np.asarray(pil_img_lr), cv2.COLOR_RGB2BGR)
        img_lr = np.asarray(pil_img_lr)

        hr = torch.squeeze(hr, dim=0)
        pil_img_hr = transforms.ToPILImage()(hr)
        # img_lr = cv2.cvtColor(np.asarray(pil_img_lr), cv2.COLOR_RGB2BGR)
        img_hr = np.asarray(pil_img_hr)

        psnr = compare_psnr(sr, img_hr)
        ssim = compare_ssim(sr, img_hr, multichannel=True)
        total_psnr += psnr
        total_ssim += ssim
        psnr_lst.append(psnr)
        ssim_lst.append(ssim)
        mse_physical_par_lst.append((inf_phy_par.to("cpu") - phy_par) ** 2)

        sr = np.hstack([sr, img_lr, img_hr])
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", sr)
        cv2.waitKey(0)
        # if i > 10:
        #     break

    print("avaerage psnr = ", sum(psnr_lst) / len(psnr_lst))
    print("avaerage ssim = ", sum(ssim_lst) / len(ssim_lst))
    print("average mse phy_par = ", sum(mse_physical_par_lst) / len(mse_physical_par_lst))
    # plt.figure()
    # plt.scatter(psnr_lst, mse_physical_par_lst, c="r")
    # plt.scatter(ssim_lst, mse_physical_par_lst, c="b")
    # plt.show()

