import os
import numpy as np

root_annotation_path = "/home/smj/DataSet/DAVIS2017/Annotations/"

expr_individual_path = os.path.join(root_annotation_path, "expr_feat_individual")
expr_ave_path = os.path.join(root_annotation_path, "expr_feat_ave")

sequence_name_list = os.listdir(expr_individual_path)
sequence_name_list.sort()


for sequence_name in sequence_name_list:
  sequenc_expr_individual_path = os.path.join(expr_individual_path, sequence_name)
  target_expr_name_list = os.listdir(sequenc_expr_individual_path)
  target_num = len(target_expr_name_list)

  target_expr = np.load(os.path.join(sequenc_expr_individual_path, target_expr_name_list[0]))
  target_expr = np.reshape(target_expr, (1, 1024))

  for target_id in range(1, target_num):
    target_expr_path = os.path.join(sequenc_expr_individual_path, target_expr_name_list[target_id])
    npy = np.load(target_expr_path)
    npy = np.reshape(npy, (1, 1024))
    target_expr = np.concatenate((target_expr, npy), axis=0)

  # do the average pooling
  expr_feat_ave = np.zeros((1, 1024))
  for i in range(1024):
    sum = 0
    for target_id in range(target_num):
      sum += target_expr[target_id, i]
    ave = sum / target_num
    expr_feat_ave[0, i] = ave

  sequence_expr_ave_path = os.path.join(expr_ave_path, sequence_name+".npy")
  np.save(sequence_expr_ave_path, expr_feat_ave)




pass
