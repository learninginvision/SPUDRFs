import os
import train
import predict
from picksamples import PickSamples

pace_percent = [0.5, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18]
alpha=[15, 15, 15, 15, 15, 15, 15, 15, 15, 15]

Exp = 1
max_step = 40000 
picksamples = PickSamples(exp=Exp, percent=pace_percent, pace=0, alpha=alpha, 
                        ent_threshold=-3.0, diff_threshold=1000, ent_pick_per=1155, 
                        random_pick=False, soft=True, root='.',max_step=max_step)

for pace in range(0, len(alpha)+1):
    print('Pace %d' % pace)
    left_txt, pick_txt = picksamples.pick(pace=pace)
    print('left_txt: %s' % left_txt)
    print('pick_txt: %s' % pick_txt)
    
    train_dict = {}
    train_dict['pace'] = pace
    train_dict['record'] = str(pace) + 'VGG.record'
    train_dict['data'] = str(pace) + 'VGG'

    train_dict['save'] = './checkpoints/M' + str(Exp) 
    train_dict['tmp_dir'] = './tmp/Exp' + str(Exp) 

    if pace == 0:
        train_dict['traintxt'] = left_txt
        train_dict['base_weights'] = './model/VGG_FACE.caffemodel'
    elif pace == 1: 
        train_dict['traintxt'] = pick_txt
        train_dict['base_weights'] = './model/VGG_FACE.caffemodel'
    else:
        train_dict['traintxt'] = pick_txt

        train_dict['base_weights'] = os.path.join('./checkpoints/M' + str(Exp), str(pace-1) + 'VGG_iter_{}.caffemodel'.format(max_step)) # Changed by xgtu

    train_dict['testtxt'] = './images/MORPH-test.txt'

    print(train_dict)
    net = train.SPUDRFs(parser_dict=train_dict)
    net.train()
    with open('./Entropy.txt', 'r') as f:
        lines = f.readlines()
    assert len(lines) > 2, 'train entropy.txt is null!'
    if not os.path.isdir('./entropy/train/'):
        os.makedirs('./entropy/train/')
    fn_newEntropy = './entropy/train/' + str(pace) + 'entropy.txt'
    with open(fn_newEntropy, 'w') as f:
        f.writelines(lines)

    with open(left_txt, 'r') as f:
        left_lines = f.readlines()
    if len(left_lines) > 0:
        pred_dict = {}

        pred_dict['test'] = os.path.join('./images/txt'+ str(Exp), 'trainLeft' + str(pace) + '.txt') 
        pred_dict['predict'] = os.path.join('./MAE/mae' + str(Exp), 'MAEOnTrainLeft' + str(pace) + '.txt') 
        pred_dict['deploy'] = os.path.join(train_dict['tmp_dir'], str(pace) + 'VGG-deploy.prototxt')

        pred_dict['model'] = os.path.join(train_dict['save'], str(pace) + 'VGG_iter_{}.caffemodel'.format(max_step)) 
        diff_ave = predict.Predict(pred_dict)
        with open('./Entropy.txt', 'r') as f:
            lines = f.readlines()
        fn_save_entropy = os.path.join('./entropy/E' + str(Exp), 'entropy' + str(pace) + '.txt') 
        with open(fn_save_entropy, 'w') as f:
            f.writelines(lines)

        assert len(lines) > 2, 'predict entropy.txt is null!'
