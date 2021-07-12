import os
from solver import Forest_solver
from dataset_MPII import get_loader
import math
import numpy as np 
import torch

def getAngle(label, pred):
    '''
    label: [bs, task]
    pred: [bs, task]
    '''
    def getVec(pitch, yaw):
        pitch = pitch*100.0/180.0*np.pi
        yaw = yaw*100.0/180.0*np.pi
        x = torch.cos(pitch)*torch.cos(yaw)
        y = torch.cos(pitch)*torch.sin(yaw)
        z = torch.sin(pitch)
        return torch.cat([x, y, z], dim=1)
    pitch_label = label[:, 0].reshape(-1, 1)
    yaw_label = label[:, 1].reshape(-1, 1)
    pitch_pred = pred[:, 0].reshape(-1, 1)
    yaw_pred = pred[:, 1].reshape(-1, 1)
    gaze_label = getVec(pitch_label, yaw_label) # bs, 3
    gaze_pred = getVec(pitch_pred, yaw_pred) # bs, 3
    cos = torch.clamp(torch.sum(gaze_label*gaze_pred, dim=1), -1, 1)
    angel = torch.abs(torch.acos(cos)/np.pi*180.0)
    return angel

def Predict(para_dict):
    print(para_dict)
    test_txt = para_dict['test_txt'] 
    train_txt = para_dict['train_txt']
    predict_txt = para_dict['predict_txt'] 
    pretrain_model = para_dict['pretrain_model'] 
    img_dir =  para_dict['img_dir']
    lr = para_dict['lr']
    num_trees = para_dict['num_trees']
    tree_depth = para_dict['tree_depth']
    pace = para_dict['pace']
    batch_size = 64
    num_classes = para_dict['num_classes']
    dataloader = get_loader(train_txt=train_txt, test_txt=test_txt, image_dir=img_dir, 
        batch_size=batch_size, train=False, shuffle=False)
    solver = Forest_solver(pace, pretrain_model, train_txt, 20, lr, num_trees, tree_depth, 
        num_classes=num_classes,predict=True)
    solver.load_model(pretrain_model[0], pretrain_model[1], pretrain_model[2], pretrain_model[3])

    with open(test_txt, 'r') as f:
        testlines = f.readlines()
    L = len(testlines)
    max_step = math.ceil(L*1.0 / batch_size)
    dataiter = iter(dataloader)
    
    f_pred = open(predict_txt, 'w')
    cnt = 0
    diff_sum = 0.0
    print('predicting...')
    for idx in range(max_step):
        leye, reye, label, _, headpose = next(dataiter)
        leye = leye.cuda()
        reye = reye.cuda()
        label = label.cuda().float()
        headpose = headpose.cuda().float()
        
        pred, entropy = solver.get_entorpy(leye, reye, headpose)

        angle = getAngle(label, pred)
        
        diff = angle.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        
        diff_sum += np.sum(diff)
        for i in range(headpose.shape[0]):
            limg, rimg, pitch, yaw, _, hx, hy = testlines[cnt].strip('\n').split(' ')
            img = '{} {}'.format(limg, rimg)
            headpose = '{} {}'.format(hx, hy)
            line_pred = 'img name: {}, label: p{}y{}, pred: p{}y{}, ent: {:.6f}, diff: {:.6f}, headpose: {}\n'.format(
                img, pitch, yaw, pred[i][0], pred[i][1], entropy[i], diff[i], headpose)
            f_pred.write(line_pred)
            cnt += 1

    print('pace: {}, samples: {}'.format(pace, cnt))
    if cnt > 0:
        print(diff_sum/cnt)
    f_pred.close()
