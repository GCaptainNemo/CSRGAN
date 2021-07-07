from src.data import DataSampler
from src.model import GeneratorResNet, Discriminator
import torch

from src.train import Train

txt_address = "../data/train.txt"

data_sampler = DataSampler(txt=txt_address, batch_size=5)
# generator = GeneratorResNet().cuda()
generator = GeneratorResNet().cuda()
discriminator = Discriminator().cuda()

# discriminator = Discriminator([3, 50, 50]).cuda()
train = Train(data_sampler=data_sampler, generator=generator, discriminator=discriminator,
              )
train.train(epoch_num=1, d_learning_rate=3e-4, g_learning_rate=3e-4)


with open("generator.pth", "wb+") as f:
    torch.save(generator, f)

with open("generator.pth", "wb+") as f:
    torch.save(discriminator, f)


