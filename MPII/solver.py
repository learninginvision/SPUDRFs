from networks.vggbn import VGG_16 as VGG_16
from utils.kmeans import kmeans
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import forest as ndf

import scipy.stats as st

class Forest_solver():
    '''
    :param \n
    pace: int \n
    pretrain_model:  ("vgg_face_torch/VGG_FACE.t7", pace, dataset, epoch) \n
    train_txt: str \n
    iterations_update_forest: default 20 \n
    lr, num_trees, tree_depth, num_classes, predict=False
    '''
    def __init__(self, pace, pretrain_model, train_txt, iterations_update_forest,
        lr=0.05, num_trees=5, tree_depth=6, num_classes=67, lr_policy=0, predict=False):
        self.num_trees = num_trees
        self.num_classes = num_classes
        self.leaf_node_num = 2 ** (tree_depth - 1)
        self.lr = lr
        self.lr_policy = lr_policy

        self.feat_layer = VGG_16()
        forest = ndf.Forest(n_tree=num_trees, tree_depth=tree_depth, n_in_feature=128,
            num_classes=num_classes, iterations_update_forest=iterations_update_forest)
        model = ndf.NeuralDecisionForest(self.feat_layer, forest)
        model = model.cuda()
        self.model = model
        self.weight_decay = 0.0
        self.momentum = 0.9
        
        if self.lr_policy == 0:
            param_list = self.feat_layer.get_weight_dict(lr, self.weight_decay)
            self.optimizer = torch.optim.Adam(param_list, lr=lr, weight_decay=0.0)
        if self.lr_policy == 1:
            param_list = self.feat_layer.get_weight_dict_1(lr, self.weight_decay, momentum=self.momentum)
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.5*1e-5)
        if not predict and pace < 2:
            print('load model from %s' % pretrain_model[0])
            self.feat_layer.load_weights(path=pretrain_model[0])
            init_mean, init_sigma = self.kmeans_label(train_txt)
            self.model.forest.dist.init_kmeans(init_mean, init_sigma)

    def kmeans_label(self, train_txt):
        labels = []
        with open(train_txt,"r") as f: 
            lines = f.readlines()
            for line in lines:
                if 'noise' not in line:
                    pitch = line.strip('\n').split(' ')[2]
                    yaw = line.strip('\n').split(' ')[3]
                    pitch = float(pitch)
                    yaw = float(yaw)
                    labels.append([pitch, yaw])
        labels = np.reshape(np.array(labels), [-1, 2])
        init_mean, init_sigma = kmeans(labels, self.leaf_node_num)
        return init_mean, init_sigma

    def forward(self, leye, reye, headpose):
        '''
        :return
        predictions -> (bs, tree, task) 
        pred4Pi -> (bs, leaf, tree)
        '''
        predictions, pred4Pi, feat = self.model(leye, reye, headpose)

        return predictions, pred4Pi, feat

    def get_loss(self, leye, reye, y, weight, headpose):
        predictions, pred4Pi, features = self.forward(leye, reye, headpose)
        y = y.reshape(-1, 2).unsqueeze(1).repeat(1, self.num_trees, 1)  # bs, tree, task
        weight = weight.reshape(-1, 1).unsqueeze(1).repeat(1, self.num_trees, 2).float()/10000.0 # bs, tree, task
        loss = torch.sum(0.5 * weight * (y - predictions) ** 2)/leye.shape[0]
        return loss, pred4Pi

    def test(self, leye, reye, headpose):
        self.model.eval()
        with torch.no_grad():
            pred, _, _ = self.forward(leye, reye, headpose) # bs, tree, task
            pred = torch.mean(pred, dim=1) # bs, task
            return pred

    def backward_theta(self, leye, reye, y, weight, headpose):
        self.model.train()
        self.optimizer.zero_grad()
        l2_loss, pred4Pi = self.get_loss(leye, reye, y, weight, headpose)
        loss = 1.0 * l2_loss
        loss.backward()
        self.optimizer.step()

        return l2_loss.item(), pred4Pi

    def backward_pi(self, x, y):
        self.model.forest.dist.update(x, y)
        
    def get_entorpy(self, leye, reye, headpose):
        # pred4Pi -> bs, 32, 5
        self.model.eval()
        with torch.no_grad():
            pred, pred4Pi, _ = self.forward(leye, reye, headpose)
            pred = torch.mean(pred, dim=1) # bs, 5, task
            pred4Pi = pred4Pi.detach().cpu().numpy().transpose((0, 2, 1)) # bs, 5, 32
            
            sigma = self.model.forest.dist.sigma # 5, 32, task, task
            entropy = np.zeros((sigma.shape[0], sigma.shape[1])) # 5, 32
            for i in range(sigma.shape[0]):
                for j in range(sigma.shape[1]):
                    entropy[i, j] = st.multivariate_normal.entropy(cov=sigma[i, j, :, :])
            entropy = np.expand_dims(entropy, 0).repeat(headpose.shape[0], 0) # bs, 5, 32
            entropy = pred4Pi * entropy
            entropy = (np.sum(entropy, axis=(1,2)) / self.num_trees).reshape(-1)
            return pred, entropy 

    def save_model(self, path, pace, dataset_name, epoch):
        print('save model at {}_model_{}.pth'.format(path + dataset_name + str(pace), epoch))
        torch.save(self.model.state_dict(), path + dataset_name + str(pace) + '_model_{}.pth'.format(epoch))

        self.model.forest.dist.save_model(path, pace, epoch)

    def load_model(self, path, pace, dataset_name, epoch):
        print('load model from {}_model_{}.pth'.format(path + dataset_name + str(pace), epoch))
        self.model.load_state_dict(torch.load(path + dataset_name + str(pace) + '_model_{}.pth'.format(epoch)))

        self.model.forest.dist.load_model(path, pace, epoch)

    def update_lr(self):
        if self.lr_policy == 0:
            self.update_lr_0()

    def update_lr_0(self):
        if self.lr > 1e-10:
            self.lr = self.lr * 0.8
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        print('Update lr, new lr = %.10f' % self.lr)
