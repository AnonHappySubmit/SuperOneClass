import random
import time
import os, sys
import math
import argparse
from collections import OrderedDict
from torch.autograd import Variable
import numpy as np
from visualize import Visualizer
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from collections import defaultdict
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score,  average_precision_score
from dataset import *
from scipy.interpolate import interpn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        #print ("self.centers is parameter, can be updated" )
        self.centerlossfunc = CenterlossFunc()
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim, feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size)
        return loss

class CenterlossFunc(nn.Module):
    def __init__(self):
        super(CenterlossFunc, self).__init__()
    def forward(self, feature, label, centers, batch_size):
        centers_batch = centers.index_select(0, label.long())
        
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

# number of gpu's available
ngpu = 1
# input noise dimension
nz = 100
# number of generator filters
ngf = 64
#number of discriminator filters
ndf = 64

nc = 1


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
class Rotation(nn.Module):
    def __init__(self, max_range = 4):
        super(Rotation, self).__init__()
        self.max_range = max_range


    def forward(self, input, aug_index):
        _device = input.device

        _, _, H, W = input.size()

        aug_index = aug_index % self.max_range
        output = torch.rot90(input, aug_index, (2, 3))
        return output
    
class CutPerm(nn.Module):
    def __init__(self, max_range = 4):
        super(CutPerm, self).__init__()
        self.max_range = max_range
        
    def _cutperm(self, inputs, aug_index):

        _, _, H, W = inputs.size()
        h_mid = int(H / 2)
        w_mid = int(W / 2)

        jigsaw_h = aug_index // 2
        jigsaw_v = aug_index % 2

        if jigsaw_h == 1:
            inputs = torch.cat((inputs[:, :, h_mid:, :], inputs[:, :, 0:h_mid, :]), dim=2)
        if jigsaw_v == 1:
            inputs = torch.cat((inputs[:, :, :, w_mid:], inputs[:, :, :, 0:w_mid]), dim=3)

        return inputs

    def forward(self, input, aug_index=None):
        _device = input.device

        _, _, H, W = input.size()


        aug_index = aug_index % self.max_range
        output = self._cutperm(input, aug_index)
        return output
    

class Model(nn.Module):
    def __init__(self, ):
        super(Model, self).__init__()
        
        self.cls_layer = nn.Sequential(nn.Conv2d(ndf * 8, 1, 2, 1, 0, bias=False),nn.Sigmoid()) # not actually used. 
        self.fc_layer = nn.Sequential(nn.Linear(128, 2))                                       # not actually used. 
        
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, ndf * 8, 2, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
        )
        
        self.feat_layer = nn.Sequential(nn.Conv2d(ndf * 8, 2, 2, 1, 0, bias = False))
        

    def forward(self, input):
        
        bottle = self.main(input)
        
        feat = self.feat_layer(bottle)
        
        return feat.squeeze()
        


class Trainer(object):
    def __init__(self, cfig):
        self.cfig = cfig
        self.device = device

        self.model = Model().to(self.device)
        self.model.apply(weights_init)
        self.Centloss = CenterLoss(cfig['num_class'], 2).to(self.device)
        
        
        self.optimzer4center = optim.Adam(self.Centloss.parameters(), lr = 0.01, betas = (.5, .999))
        self.optim = optim.Adam(self.model.parameters(), lr=cfig['lr'], betas=(.5, .999))
        
        if cfig['dataset'] == 'fmnist':
            datasets = FMNIST_Dataset(cfig['data_path'],  cfig['normal_class'])
        

        self.train_loader = torch.utils.data.DataLoader(datasets.train_set, batch_size = cfig['batch_size'], num_workers = 4)
        self.test_loader = torch.utils.data.DataLoader(datasets.test_set, batch_size = cfig['batch_size'], num_workers = 4)
        
        self.vis = Visualizer(cfig['save_path'])
    
    def get_lr(self):
        for param_group in self.optim.param_groups:
            return param_group['lr']
        
    def adjust_learning_rate(self, optimizer, base_lr, epoch):

        if epoch > 150:
            lr = base_lr * 0.1 
        elif epoch > 100:
            lr = base_lr * 0.3
        else:
            lr = base_lr

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def visualize(self, feat, labels, centers, epoch, suffix = '', act_center = None):

        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        plt.figure()
        
        label_set = set(labels)
        for lab in label_set:
            plt.plot(feat[labels == lab, 0], feat[labels == lab , 1], '.', c = c[lab])
        
        if centers is not None: 
            for i in range(len(centers)):
                plt.plot(centers[i, 0], centers[i, 1], '*', c = c[-i], markersize = 20)
            
        if act_center is not None: 
            assert isinstance(act_center, list)
            for i in range(len(act_center)):
                center = act_center[i]
                plt.plot(center[0], center[1], '*', c = c[-i], markersize = 20)

        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc = 'upper right')

        plt.text(-7.8, 7.3,"epoch=%d" % epoch)
        if not os.path.exists(self.cfig['save_path'] + '/fig'):
            os.mkdir(self.cfig['save_path'] + '/fig')
        plt.savefig(self.cfig['save_path']  + '/fig/epoch%d%s.jpg' % (epoch, suffix))
        plt.close('all')

        
    def train(self):

        for epoch in range(self.cfig['max_epoch']):

            
            self.loss_breakdown = defaultdict(float)
            #self.adjust_learning_rate(self.optim_G, self.cfig['lr_g'], epoch)
            self.adjust_learning_rate(self.optim, self.cfig['lr'], epoch)
            self.loss_breakdown['100*lr'] = 100 * self.get_lr() 
            
            model_root = os.path.join(self.cfig['save_path'], 'models')
            if not os.path.exists(model_root):
                os.mkdir(model_root)
            model_pth = '%s/model_last.pth' % (model_root)
            
            torch.save(model_dict, model_pth)
                
            feat_true_list = []
            feat_center_list = []
            normal_list, fake_list = [], []
            label_list = []
                
            for batch_idx, (x, y) in enumerate(self.train_loader):

                # data preparation
                self.model.zero_grad()
                
                x, y = Variable(x).to(self.device), y.to(self.device)
                
                x_train_normal = x 

                feat_t, out_t = self.model(x_train_normal)

                #data_rotat = torch.cat([Rotation()(x_train_normal, k) for k in range(1, 4)])
                
                data_perm = torch.cat([CutPerm()(x_train_normal, k) for k in range(1, 4)])
                
                center_label = torch.cat([torch.ones_like(y) * k for k in range(0, 4)], 0) 
                

                feat_p = self.model(data_perm)
                
                center_feat = torch.cat([feat_t,feat_p])

                
                center_loss = self.Centloss(center_label, center_feat) * center_alpha
                unicenter_loss = UniformLoss() * self.cfig['uniform_wei']  # this line is not complete
                dispcenter_loss =  DispLoss()  * self.cfig['disp_wei'] #  this line is not complete
                
                d_loss = center_loss + unicenter_loss + dispcenter_loss  
                # center_loss and dispcenter_loss is the homo-divisive loss in our paper
                # unicenter_loss is the 

                self.optim.zero_grad()
                self.optimzer4center.zero_grad()
                d_loss.backward()
                self.optim.step()
                self.optimzer4center.step()

                self.loss_breakdown['center'] += center_loss.item() 
                self.loss_breakdown['uniform'] += unicenter_loss.item() 
                self.loss_breakdown['disp'] += dispcenter_loss.item() 
                
                
                feat_center_list.append(center_feat.cpu().data)
                label_list.append(center_label.cpu().data)
                
                ### train generator
            self.loss_breakdown['center'] /= len(self.train_loader)
            self.loss_breakdown['uniform'] /= len(self.train_loader)
            self.loss_breakdown['disp'] /= len(self.train_loader)

            all_label = torch.cat(label_list, 0)

            all_feat = torch.cat(feat_center_list, 0) 
            
            self.visualize(all_feat.numpy(), all_label.numpy(), self.Centloss.centers.data.cpu().numpy(), epoch)

            self.vis.plot_loss(epoch, self.loss_breakdown)
            
    def test_epoch(self, load_path = None):
        
        
        trained_model = torch.load(load_path)
        self.model.load_state_dict(trained_model['model'])
        self.Centloss.load_state_dict(trained_model['centloss'])
        
        normal_list, rotat_list, anomly_list = [], [], []
        dist_t_list, dist_r_list, dist_a_list = [], [], []

        loader = self.test_loader

            
        feat_aug_list = []
        feat_true_list = []
        feat_anomaly_list = []

        for batch_idx, (x, y) in enumerate(loader):
            
            normal_indx = (y == 1)
            anomaly_indx = (y == 0)
            
            x_test_normal = x[normal_indx].to(self.device)
            x_test_anomaly = x[anomaly_indx].to(self.device)
            
            data_r = torch.cat([CutPerm()(x_test_normal, k) for k in range(1, 4)])
            
            pdb.set_trace()
            
            feat_t = self.model(x_test_normal)
            feat_a = self.model(x_test_anomaly)
            feat_r = self.model(data_r)
            
            dist_t = torch.norm(feat_t - self.Centloss.centers[0], dim = 1)
            dist_a = torch.norm(feat_a - self.Centloss.centers[0], dim = 1)
            dist_r = torch.norm(feat_r - self.Centloss.centers[0], dim = 1)
            
            dist_t_list += dist_t.data.cpu().numpy().tolist()
            dist_a_list += dist_a.data.cpu().numpy().tolist()
            dist_r_list += dist_r.data.cpu().numpy().tolist()

            
            feat_aug_list.append(feat_r.cpu().data)
            feat_true_list.append(feat_t.cpu().data)
            feat_anomaly_list.append(feat_a.cpu().data)
        
        all_feat = torch.cat(feat_true_list+ feat_aug_list + feat_anomaly_list, 0) 
        
        #all_label = [0] * len(dist_t_list)+ [1] * len(dist_r_list) +  [2] * len(dist_a_list) 
        #self.test_visualize(all_feat.numpy(), np.array(all_label), None, epoch, suffix = '_test')
        
        AvsN_gt = [0] * len(dist_t_list) + [1] * len(dist_a_list)
        
        
        AvsN_dist = dist_t_list + dist_a_list
        
        AUC_dist = roc_auc_score(AvsN_gt, AvsN_dist)
        
        
        print ('The AUC is: {:.3f} when the normal-class indx is {:d}'.format(AUC_dist, self.cfig['normal_class']))
        


        
if __name__ == '__main__':
    
    # a example of configure figures 
    cfig = {
        'dataset': 'fmnist',
        'data_path': './data/fashionmnist',
        
        'disp_wei': 2, 
        'batch_size': 200,
        
        'input_size': 32,
        'half_space': 0,
        
        'num_class': 4, 
        'max_epoch': 201,
        'pretrain_generator': None, 
        'save_root': 'train_log' # 
    }
    
    import argparse 
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('phase',  type = str, help = "train or test")
    parser.add_argument('--lr', default = 0, type = float, help = "lr")
    parser.add_argument('--normal_class', default = 0, type = int, help = "normal_class")
    parser.add_argument('--uniform_wei', default = 0, type = float, help = "uniform_wei")
    parser.add_argument('--test_model_path', default = '', type = str, help = "test model path, only use for test phase")
    parser.add_argument('--data_path', default = './data/fashionmnist', type = str, help = "data_path")
    
    args = parser.parse_args()

    cfig['normal_class'] = args.normal_class 
    cfig['lr'] = args.lr
    cfig['uniform_wei'] = args.uniform_wei
    cfig['data_path'] = args.data_path
    cfig['save_path'] = cfig['save_root'] + '/ssl_uniform{:.4f}_{:d}'.format(args.uniform_wei, args.normal_class) 
    print (cfig['normal_class'])
    
    trainer = Trainer(cfig)

    if args.phase == 'train':
        print (cfig['save_path'])
        trainer.train()
    
    if args.phase == 'test':
        trainer.test_epoch(load_path = args.test_model_path)

    
