import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import os
from datetime import datetime

from CFNet import CFNet
from Config import Config
from CFNetDataset import CFNetDataset


def extract_image_feature(normalization, image_path, model):
  img = Image.open(image_path)
  img = normalization(img)
  img.unsqueeze_(dim=0)
  output = model.conv1(img)
  output = model.bn1(output)
  output = model.relu(output)
  output = model.maxpool(output)
  output = model.layer1(output)
  output = model.layer2(output)
  output = model.layer3(output)
  output = model.layer4(output)
  output = model.avgpool(output)

  frame_feat = output.detach().numpy()
  frame_feat = np.reshape(frame_feat, (1, 2048))

  return frame_feat


# control the precision of the float when printing the log information
def control_precision(value, digit_num):
  value = float(value)
  value = round(value, digit_num)
  value_str = str(value)
  value_str_length = len(value_str)
  if value_str != (digit_num+3):
    for i in range(digit_num + 3 - value_str_length):
      value_str = value_str + "0"
  return value_str


def main():

  ###################################################
  ############ Inilization ##########################
  ###################################################

  # the sequences using in the training dataset
  sequence_list = Config.sequence_list

  # check the running time
  start_time = datetime.now()

  # device information
  use_cuda = torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')

  # initialize the average performaces of each sequence
  seq_frame_ave_performance = []

  for sequence_name in sequence_list:
    sequence_length = len(os.listdir(os.path.join(Config.root_image_path, sequence_name)))
    performance_path = os.path.join(Config.root_annotation_path, "language_performance", sequence_name)
    num_target = int(len(os.listdir(performance_path)))

    # for all targets, and all frame performaces for each target
    target_performace_list = []

    for target_id in range(0, num_target):
      target_performace_path = os.path.join(performance_path, sequence_name + "_" + str(target_id) + ".txt")
      f = open(target_performace_path)
      performance_list = f.read().splitlines()
      target_performace_list.append(performance_list)

    # the average performace of each frame (the average value of different targets)
    ave_performace_list = []
    for frame_id in range(sequence_length):
      sum = 0
      for target_id in range(num_target):
        sum += float(target_performace_list[target_id][frame_id])
      ave_performace_list.append(sum / num_target)

    seq_frame_ave_performance.append(ave_performace_list)


  # initialize the transform function
  normalization = transforms.Compose([
      transforms.Resize((224,224)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
  ])

  # initialize the CFNet and Resnet
  if Config.first_train == True:
    resnet50 = models.resnet50(pretrained=True)
    resnet50 = resnet50
    if Config.train_resnet == False:
      resnet50.eval()
    cfnet = CFNet().to(device)
  else:
    resnet50 = torch.load(Config.save_resnet_path)
    resnet50 = resnet50
    if Config.train_resnet == False:
      resnet50.eval()
    cfnet = torch.load(Config.save_cf_path)
    cfnet = cfnet.to(device)


  # initialize the CFNetDataset
  cfnet_dataset = CFNetDataset(normalization, resnet50, sequence_list, Config.root_image_path, Config.root_annotation_path, seq_frame_ave_performance, Config.frame_sample_num, device)
  #cfnet_dataset = CFNetDataset(normalization)

  # initialize the dataloader (batch and workers)
  train_dataloader = DataLoader(cfnet_dataset,
                          shuffle=True,
                          num_workers=8,
                          batch_size=Config.batch_num)


  optimizer = optim.Adam(cfnet.parameters(), lr=Config.lr)
  criterion = nn.MSELoss(reduce=False, size_average=False)


  ###################################################
  ############### Training ##########################
  ###################################################

  for iter in range(Config.iter_num):

    for i, data in enumerate(train_dataloader):
      input, label = data
      input = input.to(device)
      y = label.to(device)

      # prediction
      fx = cfnet(input)

      # backward
      loss = criterion(fx, y)
      loss_sum = loss.sum()
      loss_sum.backward()
      optimizer.step()

      # log information
      fx_numpy = fx.detach().cpu().numpy()
      fx_list = fx_numpy.tolist()
      y_numpy = y.detach().cpu().numpy()
      y_list = y_numpy.tolist()

      loss_numpy = loss.detach().cpu().numpy()
      loss_numpy_sum = loss_numpy.sum()

      current_time = datetime.now()
      used_time = current_time - start_time

    # print the log information (the last one in the batch )
    print("i: " + str(iter), ", loss: " + str(loss_numpy_sum) + ", time: " + str(used_time))
    for fx_value in fx_list[0]:
      print(control_precision(fx_value, 3) + ", ", end="")
    print()
    for y_value in y_list[0]:
      print(control_precision(y_value, 3) + ", ", end="")
    print()
    print()

    # save the model
    if (iter % Config.save_iters == 0) and (iter > 0):
      torch.save(cfnet, Config.save_cf_path)
      torch.save(resnet50, Config.save_resnet_path)


if __name__ == '__main__':
  main()

