import glob
import numpy as np
import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset

class trainset_loader(Dataset):
    def __init__(self, root, dose):
        self.file_path = 'input_' + dose
        self.files_A = sorted(glob.glob(os.path.join(root, 'train', self.file_path, 'data') + '*.mat'))
        
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A.replace(self.file_path,'label_single')
        file_C = file_A.replace('input','projection')
        input_data = scio.loadmat(file_A)['data']
        label_data = scio.loadmat(file_B)['data']
        prj_data = scio.loadmat(file_C)['data']
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        return input_data, label_data, prj_data

    def __len__(self):
        return len(self.files_A)

class testset_loader(Dataset):
    def __init__(self, root, dose):
        self.file_path = 'input_' + dose
        self.files_A = sorted(glob.glob(os.path.join(root, 'test', self.file_path, 'data') + '*.mat'))
        
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A.replace(self.file_path,'label_single')
        file_C = file_A.replace('input','projection')
        res_name = 'result\\' + file_A[-13:]
        input_data = scio.loadmat(file_A)['data']
        label_data = scio.loadmat(file_B)['data']
        prj_data = scio.loadmat(file_C)['data']
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        return input_data, label_data, prj_data, res_name

    def __len__(self):
        return len(self.files_A)
