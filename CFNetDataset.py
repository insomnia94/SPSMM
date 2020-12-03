from torch.utils.data import Dataset
import numpy as np
import random
import torch
import os
from Config import Config

#class CFNetDataset(Dataset):
class CFNetDataset():
  def __init__(self, normalization, resnet, sequence_list, root_image_path, root_annotation_path, seq_frame_ave_performance, frame_sample_num, device):
    self.normalization = normalization
    self.resnet = resnet.eval()
    self.sequence_list = sequence_list
    self.root_image_path = root_image_path
    self.root_annotation_path = root_annotation_path
    self.seq_frame_ave_performance = seq_frame_ave_performance
    self.frame_sample_num = frame_sample_num
    self.device = device

  def __getitem__(self, index):

    # choose a random sequence from the candicate sequence list
    num_sequence = len(self.sequence_list)

    # the id number of the sequence chosen
    sequence_id = random.randint(0, num_sequence - 1)
    sequence_name = self.sequence_list[sequence_id]

    # the list of the names of all frames in the sequence
    sequence_image_path = os.path.join(self.root_image_path, sequence_name)
    image_name_list = os.listdir(sequence_image_path)
    image_name_list.sort()

    # the number of the frames in this sequence
    sequence_length = len(image_name_list)

    # choose some random frames as the chosen frames from the whole sequeunce
    frame_compare_id_list = random.sample(range(sequence_length), self.frame_sample_num)
    frame_compare_id_list.sort()

    ##############################################
    ##### generate the appearance feature  #######
    ##############################################

    root_feature_path = os.path.join(self.root_annotation_path, "appearance_feature")
    sequence_feature_path = os.path.join(root_feature_path, sequence_name)

    # frame_feature_list is used to store the appearance feature (extracted by Resnet) for each frame
    frame_feature_list = []

    for frame_compare_id in frame_compare_id_list:
      frame_feature_path = os.path.join(sequence_feature_path, image_name_list[frame_compare_id])
      frame_feature_path = frame_feature_path + ".npy"

      frame_feature = np.load(frame_feature_path)

      frame_feature_list.append(frame_feature)

    # convert the list to tensor
    frames_feature_tensor = torch.Tensor(np.array(frame_feature_list))
    frames_feature_tensor = frames_feature_tensor.view(self.frame_sample_num * 2048)

    #################################
    ##### generate the lable     ####
    #################################

    ave_performace_list = self.seq_frame_ave_performance[sequence_id]

    # choose the average performance value we need (label)
    label_list = []
    for frame_compare_id in frame_compare_id_list:
      label_list.append(ave_performace_list[frame_compare_id])

    label = np.array(label_list)
    label = np.reshape(label, (self.frame_sample_num))
    label = torch.Tensor(label)

    #####################################
    ### extract the expression feature ##
    #####################################

    expr_feat_ave_path = os.path.join(self.root_annotation_path, "expr_feat_ave", sequence_name + ".npy")
    expr_feat_ave = np.load(expr_feat_ave_path)
    expr_feat_ave = torch.Tensor(expr_feat_ave)
    expr_feat_ave = expr_feat_ave.view(1024)

    input = torch.cat([frames_feature_tensor, expr_feat_ave], 0)

    return input, label


  def __len__(self):
    return Config.batch_num



