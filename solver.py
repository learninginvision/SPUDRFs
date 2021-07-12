from networks.vggFace import VGG_16 as VGG_16
from utils.kmeans import kmeans
import torch
import numpy as np

import forest as ndf

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
        lr=0.05, num_trees=5, tree_depth=6, num_classes=67, predict=False):
        self.num_trees = num_trees
        self.num_classes = num_classes
        self.leaf_node_num = 2 ** (tree_depth - 1)
        self.lr = lr

        self.feat_layer = VGG_16()
        forest = ndf.Forest(n_tree=num_trees, tree_depth=tree_depth, n_in_feature=2622,
            num_classes=num_classes, iterations_update_forest=iterations_update_forest)
        model = ndf.NeuralDecisionForest(self.feat_layer, forest)
        model = model.cuda()
        self.model = model
       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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
                    label = line.strip('\n').split(' ')[1]
                    label = float(label)
                    labels.append(label)
        labels = np.reshape(np.array(labels), [-1, 1])
        init_mean, init_sigma = kmeans(labels, self.leaf_node_num)
        return init_mean, init_sigma

    def forward(self, x):
        '''
        :return
        predictions -> (bs, tree)
        pred4Pi -> (bs, leaf, tree)
        '''
        predictions, pred4Pi = self.model(x)

        return predictions, pred4Pi

    def get_loss(self, x, y, weight):
        predictions, pred4Pi = self.forward(x)
        loss = torch.sum(weight.unsqueeze(1).float()/10000.0 * 0.5 * (y.view(-1, 1) - predictions) ** 2)/x.shape[0]
        return loss, pred4Pi

    def test(self, x):
        self.model.eval()
        with torch.no_grad():
            pred, _ = self.forward(x)
            return torch.mean(pred, dim=1)

    def backward_theta(self, x, y, weight):
        self.model.train()
        self.optimizer.zero_grad()
        loss, pred4Pi = self.get_loss(x, y, weight)
        
        loss.backward()
        self.optimizer.step()

        return loss.item(), pred4Pi

    def backward_pi(self, x, y):
        self.model.forest.dist.update(x, y)
        
    def get_entorpy(self, x):
        # pred4Pi -> bs, 32, 5
        self.model.eval()
        with torch.no_grad():
            pred, pred4Pi = self.forward(x)
            pred = torch.mean(pred, dim=1)
            pred4Pi = pred4Pi.detach().cpu().numpy().transpose((0, 2, 1))

            sigmma = self.model.forest.dist.sigma.reshape(self.num_trees, -1)
            sigmma = np.expand_dims(sigmma, axis=0)
            sigmma = np.repeat(sigmma, pred4Pi.shape[0], axis=0)
            entropy = 0.5 * pred4Pi * (np.log(2*np.pi*sigmma + 1e-12) + 1)
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

    def check_momentum(self):
        for param_group in self.optimizer.param_groups:
            momentum = param_group['momentum']
            break
        print('momentum: {}'.format(momentum))
    
    def update_lr(self):
        if self.lr > 1e-8:
            self.lr = self.lr * 0.5
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr
        print('Update lr, new lr = %.6f' % self.lr)


