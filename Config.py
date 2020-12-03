class Config():
  lr = 1e-7
  batch_num = 128
  save_iters = 200
  frame_sample_num = 8
  iter_num = 1000000
  first_train = False 
  train_resnet = False

  # sequence_list = ["breakdance"]
  sequence_list = ["judo", "bike-packing", "soapbox", "breakdance", "dogs-jump", "bmx-trees", "scooter-black", "gold-fish", "loading", "motocross-jump", "drift-chicane"]

  #root_image_path = "/home/smj/DataSet/DAVIS2017/JPEGImages/480p/"
  #root_annotation_path = "/home/smj/DataSet/DAVIS2017/Annotations/"

  root_image_path = "/Data_HDD/smj_data/DAVIS2017/JPEGImages/480p/"
  root_annotation_path = "/Data_HDD/smj_data/DAVIS2017/Annotations/"

  save_cf_path = "./weight/cfnet.pkl"
  save_resnet_path = "./weight/resnet.pkl"

