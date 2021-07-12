import torch
from solver import Forest_solver
from dataset import get_loader
import numpy as np 

def train(train_dict):
    test_interval = 1000
    dataset_name = train_dict['dataset']
    test_txt = train_dict['test_txt'] # 
    train_txt = train_dict['train_txt']  
    pretrain_model = train_dict['pretrain_model']
    img_dir = train_dict['img_dir']
    max_step = train_dict['max_step']  # 30001
    pace = train_dict['pace']
    exp = train_dict['exp']

    num_trees = train_dict['num_trees']
    tree_depth = train_dict['tree_depth']
    num_classes = train_dict['num_classes']
    batch_size = train_dict['batch_size']
    lr = train_dict['lr']

    checkpoint_path = train_dict['checkpoint'] 

    num_batchs_update_forest = train_dict['num_batchs_update_forest'] # 50
    iterations_update_forest = train_dict['iterations_update_forest'] # 20

    MAEOntest_txt = train_dict['MAE_savepath'] + str(pace) + 'mae.txt'

    f_solver = Forest_solver(pace, pretrain_model, train_txt, iterations_update_forest, 
            lr=lr, num_trees=num_trees, tree_depth=tree_depth, 
            num_classes=num_classes, predict=False)
    if pace > 1:
        f_solver.load_model(pretrain_model[0], pretrain_model[1], pretrain_model[2], pretrain_model[3])

    train_data = get_loader(train_txt=train_txt, test_txt=test_txt, image_dir=img_dir, 
        batch_size=batch_size)
    test_data = get_loader(train_txt=train_txt, test_txt=test_txt,image_dir=img_dir, 
        batch_size=32, train=False, shuffle=False)
    
    num_batchs_update_forest = num_batchs_update_forest
    print('num_batchs_update_forest: %d' % num_batchs_update_forest)
    
    update_leaf_count = 0

    update_leaf_pred = []
    update_leaf_label = []

    f_test = open(MAEOntest_txt, 'w')

    dataiter = iter(train_data)

    for idx in range(max_step):
        try:
            data, label, weight = next(dataiter)
        except:
            dataiter = iter(train_data)
            data, label, weight = next(dataiter)
        data = data.cuda()
        label = label.cuda()
        weight = weight.cuda()
        loss_item, pred4Pi = f_solver.backward_theta(data, label, weight)

        if (idx+1) % 10 == 0:
            print('Exp_%s, Pace_%d [%d/%d] loss: %.6f'% (exp, pace, idx+1, max_step, loss_item))
            lr = f_solver.optimizer.param_groups[0]['lr']

        if (idx+1) % 10000 == 0:
            f_solver.update_lr()

        update_leaf_pred.append(pred4Pi)
        update_leaf_label.append(label.view(-1, 1))

        update_leaf_count += 1
        # update tree
        if update_leaf_count >= num_batchs_update_forest:
            update_leaf_count = 0
            update_leaf_pred = torch.cat(update_leaf_pred, dim=0).detach().cpu().numpy().transpose((0, 2, 1))
            update_leaf_label = torch.cat(update_leaf_label, dim=0).detach().cpu().numpy()

            f_solver.backward_pi(update_leaf_pred, update_leaf_label)
            update_leaf_pred = []
            update_leaf_label = []

        # test on testset
        if (idx+1) % test_interval == 0:
            test_kl, test_mae, total_num = 0.0, 0.0, 0.0
            for idx_, (image_t, label_t, _) in enumerate(test_data):
                total_num += image_t.shape[0]   
                image_t = image_t.cuda()
                label_t = label_t.cuda()

                pred = f_solver.test(image_t)

                test_mae += torch.sum(torch.abs(label_t - pred)).item()
                test_kl += torch.sum(torch.abs(label_t-pred) < (5/100)).item()
                
            test_mae = test_mae /total_num * 100
            test_kl = test_kl/ total_num
            print('[Test set] MAE: %.6f, CS: %.6f' % (test_mae, test_kl))

            # test on trainset
            train_kl, train_mae, total_num = 0.0, 0.0, 0.0
            for _, (image_t, label_t, _) in enumerate(train_data):
                total_num += image_t.shape[0]   
                image_t = image_t.cuda()
                label_t = label_t.cuda()

                pred = f_solver.test(image_t)

                train_mae += torch.sum(torch.abs(label_t - pred)).item()
                train_kl += torch.sum(torch.abs(label_t-pred) < (5/100)).item()

            train_mae = train_mae /total_num * 100
            train_kl = train_kl/ total_num
            print('[Train set]: %.6f, CS: %.6f' % (train_mae, train_kl))

            f_test.write("train MAE: %.6f, train CS: %.6f, test MAE: %.6f, test CS: %.6f\n" % 
                (train_mae, train_kl, test_mae, test_kl))

        if (idx+1)%max_step == 0:
            f_solver.save_model(checkpoint_path, pace, dataset_name, 100)

    f_test.close()
