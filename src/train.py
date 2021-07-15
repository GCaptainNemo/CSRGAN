import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import pickle
device = torch.device("cuda:0")


class Train:
    def __init__(self, data_sampler, generator, discriminator):
        self.data_sampler = data_sampler
        self.generator = generator
        self.discriminator = discriminator
        self.data_loader = DataLoader(dataset=data_sampler, batch_size=data_sampler.batch_size, shuffle=True)
        # pass label = 1
        # no pass label = 0
        self.true_label = torch.ones([self.data_sampler.batch_size, 1]).cuda(device)
        self.false_label = torch.zeros([self.data_sampler.batch_size, 1]).cuda(device)

    def train(self, epoch_num, d_learning_rate, g_learning_rate):
        optimizer_discriminator = optim.Adam(self.discriminator.parameters(), d_learning_rate)
        optimizer_generator = optim.Adam(self.generator.parameters(), g_learning_rate)
        criterion = nn.BCELoss()
        mse_loss = nn.MSELoss(reduce=True, size_average=True)
        d_loss_lst = []
        g_loss_lst = []
        for epoch in range(1, epoch_num + 1):
            # train_bar = tqdm(self.dataloader)
            # running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
            d_loss_avg = 0
            g_loss_avg = 0
            # for i, (true_hr_par_lr, gen_input_lr) in enumerate(train_bar):
            for i, (true_hr, physical_tensor, lr_img) in enumerate(self.data_loader):
                physical_tensor = physical_tensor.cuda(device)
                true_hr = true_hr.cuda(device)
                lr_img = lr_img.cuda(device)

                ############################
                # (1) Update D network: maximize D(x)-1-D(G(z))
                ###########################
                fake_hr, fake_physical_par = self.generator(lr_img)
                optimizer_discriminator.zero_grad()
                real_out = self.discriminator(true_hr)
                fake_out_1 = self.discriminator(fake_hr)

                d_loss = criterion(real_out, self.true_label) + \
                         criterion(fake_out_1, self.false_label)

                d_loss.backward()
                print("epoch = ", epoch, "iterations = ", i, "d-loss = ", d_loss.detach().item())

                d_loss_avg += d_loss.detach().item()
                optimizer_discriminator.step()

                ############################
                # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
                ###########################
                fake_hr_g, fake_phy_tensor = self.generator(lr_img)
                fake_out_1 = self.discriminator(fake_hr_g)
                #######################################

                optimizer_generator.zero_grad()
                g_loss = criterion(fake_out_1, self.true_label) + \
                           mse_loss(physical_tensor, fake_phy_tensor) + \
                        10000 * mse_loss(true_hr, fake_hr_g)
                g_loss.backward()
                optimizer_generator.step()
                g_loss_avg += g_loss.detach().item()
                print("epoch = ", epoch, "iterations = ", i, "g-loss = ", g_loss.detach().item())

            print("{} epoch: d_loss_avg = ".format(epoch), d_loss_avg / i)
            print("{} epoch:g_loss_avg = ".format(epoch), g_loss_avg / i)
            d_loss_lst.append(d_loss_avg / i)
            g_loss_lst.append(g_loss_avg / i)
            if epoch % 100 == 99:
                with open("generator-{}.pth".format(epoch), "wb+") as f:
                    torch.save(self.generator, f)

                with open("discriminator-{}.pth".format(epoch), "wb+") as f:
                    torch.save(self.discriminator, f)

        with open("d_loss.pkl", "wb+") as f:
            pickle.dump(d_loss, f)

        with open("g_loss.pkl", "wb+") as f:
            pickle.dump(g_loss, f)



