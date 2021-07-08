from src.data import DataSamplerTrain
from src.model import GeneratorResNet, Discriminator
import torch

from src.train import Train

txt_address = "../data/train.txt"

data_sampler = DataSamplerTrain(txt=txt_address, batch_size=5)
# generator = GeneratorResNet()
generator = GeneratorResNet().cuda()
# discriminator = Discriminator()
discriminator = Discriminator().cuda()

train = Train(data_sampler=data_sampler, generator=generator, discriminator=discriminator,
              )
train.train(epoch_num=10000, d_learning_rate=3e-4, g_learning_rate=3e-4)


with open("generator.pth", "wb+") as f:
    torch.save(generator, f)

with open("discriminator.pth", "wb+") as f:
    torch.save(discriminator, f)


