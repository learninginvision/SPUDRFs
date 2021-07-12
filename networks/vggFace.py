import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import pickle
class VGG_16(nn.Module):
    """
    Main Class
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()
        self.block_size = [2, 2, 3, 3, 3]
        self.conv_1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.conv_1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv_2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv_2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv_3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv_3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv_4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.conv_4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.conv_5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.bn6 = nn.BatchNorm1d(4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.bn7 = nn.BatchNorm1d(4096)
        self.fc8 = nn.Linear(4096, 2622)
    
    def load_weights(self, path="/root/data/chengshen/DRFs/vgg_face_torch/VGG_FACE.t7"):
        """ Function to load luatorch pretrained
        Args:
            path: path for the luatorch pretrained
        """
        model = torchfile.load(path)
        # print((type(model)))
        # print(model.modules)
        counter = 1
        block = 1
        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    self_layer = getattr(self, "conv_%d_%d" % (block, counter))
                    counter += 1
                    if counter > self.block_size[block - 1]:
                        counter = 1
                        block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc%d" % (block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]

    def forward(self, x):
        """ Pytorch forward
        Args:
            x: input image (224x224)
        Returns: class logits
        """
        x = F.relu(self.conv_1_1(x))
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.relu(self.conv_3_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_4_1(x))
        x = F.relu(self.conv_4_2(x))
        x = F.relu(self.conv_4_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv_5_1(x))
        x = F.relu(self.conv_5_2(x))
        x = F.relu(self.conv_5_3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)

        x = self.fc6(x)
        # x = self.bn6(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5, self.training)

        x = self.fc7(x)
        # x = self.bn7(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5, self.training)

        return self.fc8(x)

    def get_weight_dict_1(self, lr, decay, momentum=0.9):
        param_list = []
        name_list = []
        for idx, (name, param) in enumerate(self.named_parameters()):
            # print(idx, name)
            # conv1
            if idx < 4:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':10*lr, 'weight_decay': 10*decay}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':20*lr, 'weight_decay': 0}
                    name_list.append(name)
            # conv2, conv3
            if idx > 3 and idx < 14:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':10*lr, 'weight_decay': 1*decay}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':20*lr, 'weight_decay': 0}
                    name_list.append(name)
            # conv4, conv5
            if idx > 13 and idx < 26:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':lr, 'weight_decay': decay}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':2*lr, 'weight_decay': 0}
                    name_list.append(name)
            # fc6, fc7
            if idx > 25 and idx < 30:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':10*lr, 'weight_decay': 1*decay}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':20*lr, 'weight_decay': 0}
                    name_list.append(name)
            # fc8
            if idx == 30 or idx == 31:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':lr, 'weight_decay': decay}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':2*lr, 'weight_decay': 0}
                    name_list.append(name)

            print('name:', name, 'lr:', tmp_dict['lr'], 'weight_decay:', tmp_dict['weight_decay'])
            param_list.append(tmp_dict)
        return param_list

    def get_weight_dict_2(self, lr, decay, momentum=0.9):
        param_list = []
        name_list = []
        for idx, (name, param) in enumerate(self.named_parameters()):
            # print(idx, name)
            # conv1
            if idx < 4:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':10*lr, 'weight_decay': 10*decay, 'momentum': momentum/(10*lr)}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':20*lr, 'weight_decay': 0, 'momentum': momentum/(20*lr)}
                    name_list.append(name)
            # conv2, conv3
            if idx > 3 and idx < 14:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':10*lr, 'weight_decay': 1*decay, 'momentum': momentum/(10*lr)}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':20*lr, 'weight_decay': 0, 'momentum': momentum/(20*lr)}
                    name_list.append(name)
            # conv4, conv5
            if idx > 13 and idx < 26:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':lr, 'weight_decay': decay, 'momentum': momentum/(lr)}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':2*lr, 'weight_decay': 0, 'momentum': momentum/(2*lr)}
                    name_list.append(name)
            # fc6, fc7
            if idx > 25 and idx < 30:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':10*lr, 'weight_decay': 1*decay, 'momentum': momentum/(10*lr)}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':20*lr, 'weight_decay': 0, 'momentum': momentum/(20*lr)}
                    name_list.append(name)
            # fc8
            if idx == 30 or idx == 31:
                if 'weight' in name:
                    tmp_dict = {'params':param, 'lr':lr, 'weight_decay': decay, 'momentum': momentum/(lr)}
                    name_list.append(name)
                if 'bias' in name:
                    tmp_dict = {'params':param, 'lr':2*lr, 'weight_decay': 0, 'momentum': momentum/(2*lr)}
                    name_list.append(name)

            print('name:', name, 'lr:', tmp_dict['lr'], 'weight_decay:', tmp_dict['weight_decay'])
            param_list.append(tmp_dict)
        return param_list

    


if __name__ == '__main__':
    net = VGG_16()
    for name, _ in net.named_parameters():
        print(name)
    # print(net)
    
    
