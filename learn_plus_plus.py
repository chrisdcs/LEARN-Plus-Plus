# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 14:15:40 2022

@author: Chi Ding
"""

import scipy.io as scio
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
import ctlib
from skimage.metrics import peak_signal_noise_ratio as psnr

import os
from torch.utils.data import Dataset
import glob
import platform
from argparse import ArgumentParser
import numpy as np
from torch.autograd import Function

class projection(Function):
    @staticmethod
    def forward(self, input_data, options):
        # y = Ax   x = A^T y
        out = ctlib.projection(input_data, options)
        self.save_for_backward(options, input_data)
        return out

    @staticmethod
    def backward(self, grad_output):
        options, input_data = self.saved_tensors
        grad_input = ctlib.projection_t(grad_output, options)
        return grad_input, None
    
    
class projection_t(Function):
    @staticmethod
    def forward(self, input_data, options):
        # y = Ax   x = A^T y
        out = ctlib.projection_t(input_data, options)
        self.save_for_backward(options, input_data)
        return out

    @staticmethod
    def backward(self, grad_output):
        options, input_data = self.saved_tensors
        grad_input = ctlib.projection(grad_output, options)
        return grad_input, None

class Random_loader(Dataset):
    # need projection data, ground truth and input images
    def __init__(self, root, file_path, prj_file_path, sparse_view_num, train):
        self.train = train
        if train == True:
            folder = 'train'
        else:
            folder = 'test'
        
        self.file_path = file_path
        self.prj_file_path = prj_file_path
        self.sparse_view_num = sparse_view_num
        self.files = sorted(glob.glob(os.path.join(root, folder, self.file_path, 'data')+'*.mat'))
    
    def __getitem__(self, index):
        file = self.files[index]
        file_prj = file.replace(self.file_path, self.prj_file_path)
        file_label = file.replace(self.file_path, 'label_single')
        
        input_data = scio.loadmat(file)['data']
        prj_data = scio.loadmat(file_prj)['data']/3.84
        label_data = scio.loadmat(file_label)['data']
        
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        
        if self.train:
            return input_data, label_data, prj_data
        else:
            return input_data, label_data, prj_data, file[-13:]
    
    def __len__(self):
        return len(self.files)

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv1 = nn.Parameter(init.xavier_normal_(torch.Tensor(48, 1, 5, 5)))
        self.conv2 = nn.Parameter(init.xavier_normal_(torch.Tensor(48, 48, 5, 5)))
        self.conv3 = nn.Parameter(init.xavier_normal_(torch.Tensor(1, 48, 5, 5)))
        
    def forward(self, x):
        x = F.relu(F.conv2d(x, self.conv1, padding=2))
        x = F.relu(F.conv2d(x, self.conv2, padding=2))
        x = F.conv2d(x, self.conv3, padding=2)
        
        return x

class Learn_Plus_Plus(nn.Module):
    def __init__(self, iterations, phase):
        super(Learn_Plus_Plus, self).__init__()
        self.iterations = iterations
        self.INet_list = nn.ModuleList([Block() for i in range(iterations)])
        self.SNet_list = nn.ModuleList([Block() for i in range(iterations)])
        
        self.options = nn.Parameter(torch.tensor([512, 512, 256, 256, 0.006641,
                                                 0.0072, 0, 0.006134 * 2, 2.5, 2.5, 0, 0]),
                                    requires_grad=False)
        
        self.alpha_list = nn.Parameter(torch.tensor([0.]*iterations))
        self.index = nn.Parameter(torch.tensor([i*8 for i in range(64)],dtype=torch.int32),
                                  requires_grad=False)
        self.phase = phase
        
    def set_phase(self, phase):
        self.phase = phase
    
    
    def forward(self,x,prj,mask):
        sinogram_list = []
        x_list = []
        for i in range(self.phase):
            alpha = self.alpha_list[i]
            
            # radon transform input image
            prj_x = projection.apply(x,self.options)
            
            # extract f0 of the sinogram and compute residual with sparse view sinogram
            f0 = torch.index_select(prj_x,2,self.index)
            res = f0 - torch.index_select(prj,2,self.index)
            
            # upsample the residual
            res = resize(res,[512,512])
            
            sinogram = self.SNet_list[i](res)
            sinogram = sinogram * (1-mask) + res * mask
            
            gradient = projection_t.apply(sinogram, self.options)
            
            x_out = self.INet_list[i](x)
            x = x - alpha * gradient + x_out
            
            sinogram_list.append(sinogram)
            x_list.append(x)
            
        return x_list, sinogram_list
    
parser = ArgumentParser(description='Learn++')

parser.add_argument('--root', type=str, default='mayo_data_low_dose_256',help='data root directory')
parser.add_argument('--file_dir', type=str, default='fbp_64views', help='input files directory')
parser.add_argument('--file_prj_dir', type=str, default='512', help='projection files directory')
parser.add_argument('--model_dir', type=str, default='model',help='model directory')
parser.add_argument('--sparse_view_num', type=int, default=64, help='number of sparse views')
parser.add_argument('--phase_num', type=int, default=35, help='phase number')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
parser.add_argument('--start_phase', type=int, default=5, help='start phase')
# parser.add_argument('--IsContinue', type=bool, default=False, help='continue if there is saved progress')
parser.add_argument('--epochs', type=int, default = 100, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning_rate')
args = parser.parse_args()

root = args.root
file_dir = args.file_dir
file_prj_dir = args.file_prj_dir
sparse_view_num = args.sparse_view_num
phase_num = args.phase_num
epochs = args.epochs
model_dir = args.model_dir
learning_rate = args.learning_rate
start_epoch = args.start_epoch
start_phase = args.start_phase

gpu_list = '0'
if start_epoch > 1 or start_phase > 5:
    IsContinue = True
else:
    IsContinue = False

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

print('Load Data...')
if (platform.system() == 'Windows'):
    rand_loader = DataLoader(dataset=Random_loader(root, file_dir, file_prj_dir, sparse_view_num, True), 
                             batch_size=1, num_workers=0,shuffle=True)
else:
    rand_loader = DataLoader(dataset=Random_loader(root, file_dir, file_prj_dir, sparse_view_num, True), 
                             batch_size=1, num_workers=8,shuffle=True)

    def work(start_epoch, start_phase, learning_rate, IsContinue):
        model = Learn_Plus_Plus(phase_num, start_phase)
        model = nn.DataParallel(model)
        model.to(device)
        
        if IsContinue:
            model.load_state_dict(torch.load("./%s/net_params.pkl" % (model_dir), 
                                             map_location=device))
            max_p = np.load('max_psnr.npy')[0]
        else:
            np.save('max_psnr.npy', np.array([0]))
            max_p = 0
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        f_mask = torch.zeros((1,1,sparse_view_num*8,sparse_view_num*8))
        for i in range(sparse_view_num):
            f_mask[:,:,i*8,:] = 1
        f_mask = f_mask.to(device)
        
        
        for phase_i in range(start_phase, phase_num+1, 10):
            model.module.set_phase(phase_i)
            for epoch_i in range(start_epoch+1, epochs+1):
                progress = 0
                p_list = []
                e_list = []
                for _, data in enumerate(rand_loader):
                    input_data, label_data, prj_data = data
                    progress += 1
                    
                    # prepare data
                    input_data = input_data.to(device)
                    label_data = label_data.to(device)
                    prj_data = prj_data.to(device)
                    
                    x_list, prj_list = model(input_data,prj_data,f_mask)
                    x_output = x_list[-1]
                    
                    # compute and print loss
                    loss_all = torch.sum(torch.pow(x_output-label_data,2))
                    
                    if epoch_i > 1 and loss_all.item() >= 500:
                        print('diverge decay learning rate and start over \n\n')
                        learning_rate = learning_rate * 0.7
                        return epoch_i-1, phase_i, learning_rate, True
                    
                    optimizer.zero_grad()
                    loss_all.backward()
                    optimizer.step()
                    
                    output = x_output[0].detach().cpu().numpy().squeeze()
                    label = label_data[0].detach().cpu().numpy().squeeze()
                    p = psnr(output.astype(label.dtype).clip(0,1), label)
                    p_list.append(p)
                    e_list.append(loss_all.item())
                    
                    if progress % 10 == 0:
                        output_data = "[Epoch %02d/%02d] [Phase %02d/%02d] Total Loss: %.4f" % \
                            (epoch_i, epochs, phase_i, phase_num, loss_all.item()) \
                            + "\t progress: %02f" % (progress / 400 * 100) \
                            + '%\t avg psnr: ' + str(p) + "\n"
                        print(output_data)
                        #print(model.module.alpha_list)
            
                
                # save the parameters
                avg_p = np.mean(p_list)
                if avg_p > max_p:
                    max_p = avg_p
                    np.save('max_psnr.npy',np.array([max_p]))
                    torch.save(model.state_dict(), "./%s/net_params.pkl" % (model_dir))
                    print('the new max avg PSNR is: ', max_p, '\n\n')
                else:
                    print('avg PSNR is: ', avg_p, '\n\n')
                print('avg loss is: ', np.mean(e_list))
                    
            start_epoch = 0
        
        return epoch_i, phase_i, learning_rate, True

def train(start_phase, start_epoch, learning_rate, IsContinue):
    while start_epoch <= epochs and start_phase <= phase_num:
        
        start_epoch, start_phase, learning_rate, IsContinue = work(start_epoch, start_phase, 
                                                               learning_rate, IsContinue)
        

train(start_phase, start_epoch, learning_rate, IsContinue)
