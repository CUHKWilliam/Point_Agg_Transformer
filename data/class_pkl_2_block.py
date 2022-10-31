import os
import pickle
import glob
from tqdm import tqdm
import numpy as np

data_root = "../../data/geoformer/data/"
dataset = "scannetv2"
# class2scans_file = os.path.join(data_root, dataset, "class2scans.pkl")
# with open(class2scans_file, "rb") as f:
#     class2scans_scenes = pickle.load(f)
#
block_dir = os.path.join(data_root, dataset, "scenes/blocks_bs3_s3/data")
room_dir = os.path.join(data_root, dataset, "scenes")
# scans2block_dict = {}
# for class_id in class2scans_scenes.keys():
#     if class_id < 2:
#         continue
#     scans = class2scans_scenes[class_id]
#     scan_blocks = []
#     for scan in tqdm(scans):
#         files = glob.glob("{}/{}*".format(block_dir, scan))
#         filtered_files = []
#         for file in files:
#             pcs = np.load(file)
#             sem_id = pcs[:, -2]
#             inst_id = pcs[:, -1][sem_id == class_id]
#             inst_id_uniq = np.unique(inst_id)
#             inst_cnt = []
#             for a_inst_id in inst_id_uniq:
#                 if a_inst_id == -100:
#                     continue
#                 inst_cnt.append((inst_id == a_inst_id).sum())
#             if len(inst_cnt) == 0:
#                 continue
#             block_file_name = os.path.basename(file)
#             file_name = "{}.npy".format(block_file_name.split("_block")[0])
#             room_path = os.path.join(room_dir, file_name)
#             pcs_room = np.load(room_path)
#             for i, a_inst_id in enumerate(inst_id_uniq):
#                 ratio = inst_cnt[i] / float((pcs_room[:, -1] == a_inst_id).sum())
#                 if ratio > 0.7:
#                     filtered_files.append(file)
#                     break
#         filtered_files = [os.path.splitext(os.path.basename(x))[0] for x in filtered_files]
#         scan_blocks += filtered_files
#     scans2block_dict[class_id] = scan_blocks
# with open(os.path.join(data_root, dataset, "class2scans_block.pkl"), "wb") as f:
#     pickle.dump(scans2block_dict, f)

for idx in ["0", "1"]:
    comb_file = os.path.join(data_root, dataset, "test_combinations_fold{}.pkl".format(idx))
    with open(comb_file, "rb") as f:
        combs = pickle.load(f)
    combs_updated = {}

    for scene in tqdm(combs.keys()):
        comb = combs[scene]
        activate_labels = comb["active_label"]
        files = glob.glob("{}/{}*".format(block_dir, scene))
        for file in files:
            data = np.load(file)
            class_ids = np.unique(data[:, -2])
            activate_labels_updated = []
            for class_id in class_ids:
                if int(class_id) in activate_labels:
                    activate_labels_updated.append(int(class_id))
            a_comb = {
                "active_label": activate_labels_updated
            }
            for class_id in class_ids:
                if int(class_id) in comb.keys():
                    a_comb[int(class_id)] = comb[class_id]

            combs_updated[os.path.splitext(os.path.basename(file))[0]] = a_comb
    import ipdb;ipdb.set_trace()
    with open(os.path.join(data_root, dataset, "test_combinations_fold{}_block.pkl".format(idx)), 'wb') as f:
        pickle.dump(combs_updated, f)
