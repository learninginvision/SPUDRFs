from solver import Forest_solver
from dataset import get_loader
import math
import numpy as np 
import torch
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
    batch_size = 32
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
    for idx in range(max_step):
        data, label, _ = next(dataiter)
        data = data.cuda()
        label = label.cuda()
 
        pred, entropy = solver.get_entorpy(data)

        diff = torch.abs(label - pred).detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        
        diff_sum += np.sum(diff)
        for i in range(data.shape[0]):
            img, lbl, _ = testlines[cnt].strip('\n').split(' ')
            line_pred = 'img name: {}, label: {}, pred: {}, ent: {:.6f}, diff: {:.6f}\n'.format(
                img, lbl, pred[i], entropy[i], diff[i])
            f_pred.write(line_pred)
            cnt += 1
            print(line_pred.strip('\n'))
 
    print('pace: {}, samples: {}'.format(pace, cnt))
    if cnt > 0:
        print(diff_sum/cnt)
    f_pred.close()
    