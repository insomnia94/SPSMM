import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

a = torch.rand(3, 4)
b = torch.rand(3, 4)


c1 = a + b
print(c1.shape, c2.shape)
print(torch.all(torch.eq(c1, c2)))

