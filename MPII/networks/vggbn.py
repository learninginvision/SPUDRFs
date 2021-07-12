import torch
import torch.nn as nn
import torch.nn
import torch.nn.functional as F

class BackBone(nn.Module):
    def __init__(self):
        super(BackBone, self).__init__()
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.bn_1_1 = nn.BatchNorm2d(64)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn_1_2 = nn.BatchNorm2d(64)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.bn_2_1 = nn.BatchNorm2d(128)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn_2_2 = nn.BatchNorm2d(128)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.bn_3_1 = nn.BatchNorm2d(256)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn_3_2 = nn.BatchNorm2d(256)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn_3_3 = nn.BatchNorm2d(256)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.bn_4_1 = nn.BatchNorm2d(512)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_4_2 = nn.BatchNorm2d(512)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_4_3 = nn.BatchNorm2d(512)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_5_1 = nn.BatchNorm2d(512)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_5_2 = nn.BatchNorm2d(512)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.bn_5_3 = nn.BatchNorm2d(512)
        self.fc6 = nn.Linear(3072, 512)
        self.bn6 = nn.BatchNorm1d(512)
    
    def load_weights(self, path='E:/models/vgg16Conv.pth'):
        model_dict = torch.load(path)
    
        pretrain_dict = {}
        for key, val in model_dict.items():
            if 'conv' in key:
                pretrain_dict[key] =val
        net_dict = self.state_dict()
        net_dict.update(pretrain_dict)
        self.load_state_dict(net_dict)

    def forward(self, x):
        x = F.relu(self.bn_1_1(self.conv_1_1(x)), inplace=True)
        x = F.relu(self.bn_1_2(self.conv_1_2(x)), inplace=True)
        x = F.max_pool2d(x, 1, 1)
        x = F.relu(self.bn_2_1(self.conv_2_1(x)), inplace=True)
        x = F.relu(self.bn_2_2(self.conv_2_2(x)), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn_3_1(self.conv_3_1(x)), inplace=True)
        x = F.relu(self.bn_3_2(self.conv_3_2(x)), inplace=True)
        x = F.relu(self.conv_3_3(x), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn_4_1(self.conv_4_1(x)), inplace=True)
        x = F.relu(self.bn_4_2(self.conv_4_2(x)), inplace=True)
        x = F.relu(self.bn_4_3(self.conv_4_3(x)), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn_5_1(self.conv_5_1(x)), inplace=True)
        x = F.relu(self.bn_5_2(self.conv_5_2(x)), inplace=True)
        x = F.relu(self.bn_5_3(self.conv_5_3(x)), inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.bn6(x)
        x = F.relu(x, inplace=True)
        return x

class VGG_16(nn.Module):
    def __init__(self):
        super(VGG_16, self).__init__()
        self.backbone1 = BackBone()
        self.backbone2 = BackBone()

        self.fc7 = nn.Linear(512+512, 512)
        self.bn7 = nn.BatchNorm1d(512)
        self.fc8 = nn.Linear(512+2, 256)
        self.bn8 = nn.BatchNorm1d(256)
        self.fc9 = nn.Linear(256, 128)
    
    def load_weights(self, path='E:/models/vgg16Conv.pth'):
        self.backbone1.load_weights(path)
        self.backbone2.load_weights(path)

    def forward(self, leye, reye, y):
        """ Pytorch forward
        Args:
            x: input image (224x224)
            y: haed pose (N*3)
        Returns: class logits
        """
        lfeat = self.backbone1(leye)
        rfeat = self.backbone2(reye)
        x = torch.cat((lfeat, rfeat), dim=1)

        x = self.fc7(x)
        x = self.bn7(x)
        x = F.relu(x, inplace=True)

        x = torch.cat((x, y), dim=1)

        x = self.fc8(x)
        x = self.bn8(x)
        x = F.relu(x, inplace=True)

        return self.fc9(x), x

    def get_weight_dict(self, lr=3*1e-5, weight_decay=0.0):
        param_list = []
        for idx, (name, param) in enumerate(self.named_parameters()):
            if 'fc' in name and 'weight' in name:
                tmp_dict = {'params':param, 'lr':lr, 'weight_decay': weight_decay} 
            else:
                tmp_dict = {'params':param, 'lr':lr, 'weight_decay': 0.0}
            param_list.append(tmp_dict)
        return param_list

if __name__ == '__main__':

    net = VGG_16()
    net.get_weight_dict()

    # _left_model = models.vgg16(pretrained=False)
   

    # # remove the last ConvBRelu layer
    # _left_modules = [module for module in _left_model.features]
    # _left_modules.append(_left_model.avgpool)
    # left_features = nn.Sequential(*_left_modules)
    # print(left_features)

    # net = VGG_16()
    # x = torch.ones((2, 3, 36, 60))
    # headpose = torch.ones((2, 2))
    # y, _ = net(x, x, headpose)
    # print(y.shape)
    

    # idx2Conv = {'0': 'conv_1_1', '2': 'conv_1_2',
    #             '5': 'conv_2_1', '7': 'conv_2_2',
    #             '10': 'conv_3_1', '12': 'conv_3_2', '14': 'conv_3_3',
    #             '17': 'conv_4_1', '19': 'conv_4_2', '21': 'conv_4_3',
    #             '24': 'conv_5_1', '26': 'conv_5_2', '28': 'conv_5_3'}

    # path = 'E:/models/vgg16-397923af.pth'
    # net = VGG_16()
    # net.load_state_dict(torch.load('E:/models/vgg16Conv.pth'))
    # # print(net)
    # net_dict = net.state_dict()
    # for key, val in net_dict.items():
    #     print(key)
    #     print(val)
    #     break
    
    # vgg = models.vgg16(pretrained=False)
    # # # print(vgg)
   
    # model_dict = torch.load(path)
    # vgg.load_state_dict(model_dict)
    # for key, val in model_dict.items():
    #     print(key)
        
    # pretrain_dict = {}
    # for key, val in model_dict.items():
    #     if 'features' in key:
    #         idx, w = key.split('.')[1], key.split('.')[2]
    #         conv = idx2Conv[idx]
    #         new_key = '{}.{}'.format(conv, w)
    #         pretrain_dict[new_key] = val
    # net_dict.update(pretrain_dict)
    # net.load_state_dict(net_dict)
    # torch.save(net.state_dict(), './vgg16Conv.pth')
