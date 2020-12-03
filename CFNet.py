import torch
import torch.nn as nn
from Config import Config

class CFNet(nn.Module):
  def __init__(self,):
    super().__init__()

    self.liner1 = nn.Linear(17408, 1024)
    self.dropout1 = torch.nn.Dropout(p=0.1)
    self.liner2 = nn.Linear(1024, 256)
    self.dropout2 = torch.nn.Dropout(p=0.1)
    self.liner3 = nn.Linear(256, 128)
    self.dropout3 = torch.nn.Dropout(p=0.1)
    self.liner4 = nn.Linear(128, 64)
    self.classifier = nn.Linear(64, Config.frame_sample_num)

  def forward(self, x):
    x = self.liner1(x)
    #x = self.dropout1(x)
    x = self.liner2(x)
    #x = self.dropout2(x)
    x = self.liner3(x)
    #x = self.dropout3(x)
    x = self.liner4(x)
    x = self.classifier(x)
    x = torch.sigmoid(x)

    return x