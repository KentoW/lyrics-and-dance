# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.utils.data as data
import json
import joblib
import numpy as np



class Dataset(data.Dataset):
    def __init__(self, data_path, lpe_path, use_skeletal_feature, use_affective_feature):
        self.lpe = joblib.load(lpe_path)
        self.idx2id = []
        self.idx2length = []
        self.idx2skeletals = []
        self.idx2affectives = []
        if use_skeletal_feature == 0 and use_affective_feature == 0:
            exit(1)

        for strm in open(data_path, "r"):
            js_dat = json.loads(strm.strip())
            song_id = js_dat["song_id"]
            motion_data = joblib.load(js_dat["motion"])
            for motion_bar_data in motion_data:
                start_frame = motion_bar_data["start_frame"]
                skeletal = motion_bar_data["skeletal_feature"]
                affective = motion_bar_data["affective_feature"]
                self.idx2id.append((song_id, start_frame))
                if use_skeletal_feature == 0:
                    self.idx2skeletals.append(np.zeros_like(skeletal[:]))
                else:
                    self.idx2skeletals.append(skeletal[:])
                if use_affective_feature == 0:
                    self.idx2affectives.append(np.zeros_like(affective[:]))
                else:
                    self.idx2affectives.append(affective[:])
                self.idx2length.append(skeletal.shape[0])

    def __getitem__(self, idx):
        length = self.idx2length[idx]
        skeletals = self.idx2skeletals[idx]
        affectives = self.idx2affectives[idx]
        lpe = self.lpe
        lpe = np.tile(lpe, (length, 1, 1))
        return idx, torch.Tensor(skeletals), torch.Tensor(affectives), torch.Tensor(lpe.copy()), length

    def __len__(self):
        return len(self.idx2length)

def collate_fn(data):
    idxs, _skeletals, _affectives, _lpe, length = zip(*data)
    skeletals = torch.cat(_skeletals, dim=0)
    affectives = torch.cat(_affectives, dim=0)
    lpe = torch.cat(_lpe, dim=0)
    return idxs, skeletals, affectives, lpe, length









