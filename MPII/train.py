import torch
from solver import Forest_solver
from dataset_MPII import get_loader
import numpy as np 
import yaml

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

def train(train_dict):
    dataset_name = train_dict['dataset']
    test_txt = train_dict['test_txt'] # 
    train_txt = train_dict['train_txt']  
    pretrain_model = train_dict['pretrain_model']
    img_dir = train_dict['img_dir']
    max_epoch = train_dict['max_step']  # 30001
    pace = train_dict['pace']
    lr_policy = train_dict['lr_policy']
    batch_size = train_dict['batch_size']

    decayEpoch = [3]
    num_trees = train_dict['num_trees']
    tree_depth = train_dict['tree_depth']
    num_classes = train_dict['num_classes']
    lr = train_dict['lr']

    checkpoint_path = train_dict['checkpoint'] 

    iterations_update_forest = train_dict['iterations_update_forest'] # 20

    MAEOntest_txt = train_dict['MAE_savepath'] + str(pace) + 'mae.txt'

    f_solver = Forest_solver(pace, pretrain_model, train_txt, iterations_update_forest, 
            lr=lr, lr_policy=lr_policy, num_trees=num_trees, tree_depth=tree_depth, 
            num_classes=num_classes, predict=False)
    if pace > 1:
        f_solver.load_model(pretrain_model[0], pretrain_model[1], pretrain_model[2], pretrain_model[3])

    train_data = get_loader(train_txt=train_txt, test_txt=test_txt, image_dir=img_dir, 
        batch_size=batch_size)
    test_data = get_loader(train_txt=train_txt, test_txt=test_txt,image_dir=img_dir, 
        batch_size=128, train=False, shuffle=False)

    update_leaf_pred = []
    update_leaf_label = []

    f_test = open(MAEOntest_txt, 'w')
    max_step = len(train_data)

    for epoch in range(max_epoch):
        for step, (leye, reye, label, weight, headpose) in enumerate(train_data):
            leye = leye.cuda()
            reye = reye.cuda()
            label = label.cuda().float()
            weight = weight.cuda()
            headpose = headpose.cuda().float()
            l2_loss_item, pred4Pi = f_solver.backward_theta(leye, reye, label, weight, headpose)

            if (step+1) % 100 == 0:
                print('%s, Pace_%d, Epoch[%d/%d], Step[%d/%d] l2_loss: %.6f' % 
                (train_dict['exp'], pace, epoch, max_epoch, step+1, max_step, l2_loss_item,))

                lr = f_solver.optimizer.param_groups[0]['lr']
            update_leaf_pred.append(pred4Pi)
            update_leaf_label.append(label.view(-1, 2))

        # update tree
        update_leaf_pred = torch.cat(update_leaf_pred, dim=0).detach().cpu().numpy().transpose((0, 2, 1))
        update_leaf_label = torch.cat(update_leaf_label, dim=0).detach().cpu().numpy()
        f_solver.backward_pi(update_leaf_pred, update_leaf_label)
        update_leaf_pred = []
        update_leaf_label = []

        # update lr 
        if (epoch+1) in decayEpoch: #(epoch+1) % 5 == 0:
            f_solver.update_lr_0()
        print('lr = %e' % f_solver.optimizer.param_groups[0]['lr'])

        # test on testset
        test_kl, test_mae, total_num = 0.0, 0.0, 0.0
        test_pitch_mae, test_yaw_mae = 0.0, 0.0
        for _, (leye_t, reye_t, label_t, _, headpose_t) in enumerate(test_data):
            total_num += leye_t.shape[0]   
            leye_t = leye_t.cuda()
            
            reye_t = reye_t.cuda()
            label_t = label_t.cuda().float()
            headpose_t = headpose_t.cuda().float()

            pred = f_solver.test(leye_t, reye_t, headpose_t)
            angle = getAngle(label_t, pred)
            test_mae += torch.sum(angle).item()
            test_kl += torch.sum(angle < 5.0).item()

            test_pitch_mae += torch.sum(torch.abs(label_t[:,0] - pred[:,0])).item()
            test_yaw_mae += torch.sum(torch.abs(label_t[:,1] - pred[:,1])).item()
            
        test_mae = test_mae /total_num
        test_kl = test_kl/ total_num
        test_pitch_mae = test_pitch_mae / total_num * 100
        test_yaw_mae = test_yaw_mae / total_num * 100
        print('%s Pace: %d [Test set %d/%d] MAE: %.6f, CS: %.6f' % (train_dict['exp'], pace, epoch, max_epoch, 
                test_mae, test_kl))
        print('[Test set] pitch MAE: %.6f, yaw MAE: %.6f' % (test_pitch_mae, test_yaw_mae))
            
        # test on trainset
        train_kl, train_mae, total_num = 0.0, 0.0, 0.0
        # for _, (image_t, label_t, _) in enumerate(train_data):
        #     total_num += image_t.shape[0]   
        #     image_t = image_t.cuda()
        #     label_t = label_t.cuda().float()

        #     pred = f_solver.test(image_t)

        #     train_mae += torch.sum(torch.abs(label_t - pred)).item()
        #     train_kl += torch.sum(torch.abs(label_t-pred) < (5/200)).item()

        # train_mae = train_mae /total_num * 200
        # train_kl = train_kl/ total_num
        # print('%s: [Train set] MAE: %.6f, CS: %.6f' % (train_dict['exp'], train_mae, train_kl))

        f_test.write("train MAE: %.6f, train CS: %.6f, test MAE: %.6f, test CS: %.6f, pitch MAE: %.6f, yaw MAE: %.6f\n" % 
            (train_mae, train_kl, test_mae, test_kl, test_pitch_mae, test_yaw_mae))

    
        if (epoch+1)%max_epoch == 0:
            f_solver.save_model(checkpoint_path, pace, dataset_name, 80)
    f_test.close()
