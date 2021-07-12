import os
import yaml
import train as train
import predict
from picksamples import PickSamples


def checkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

PPID, moldelNum = [i for i in range(0,15)], 1 
def step(personID):
    config = {
        'exp': 74,
        'alpha': 5,
        'threshold': -5.0,
        'lr': 3*1e-5,
        'capped': False,
        'batchsize': 128,
        'ent_pick': 2000,
        'tree_depth': 6,
        'soft_percent': 0.8,
        'base_weights': '/root/new/models/vgg16Conv.pth', 
        'pace_percent': [0.5, 0.1, 0.1, 0.1, 0.1, 0.1] 
    }

    pace_percent = config['pace_percent']
    alpha=[config['alpha'] for i in range(len(pace_percent))]
    expNum = config['exp']
    person_id = personID
    direction = 'double'
    checkpointsPath = '{}/checkpoints-{}/model-{}-{}/PID-{}/'.format(r".", direction, expNum, moldelNum, personID)
    Exp = './Atp{}-{}/Exp{}'.format(expNum, direction, person_id)
    checkdir('./Atp{}-{}/'.format(expNum, direction))
    with open('./Atp{}-{}/config.yaml'.format(expNum, direction), 'w') as f:
        f.write(yaml.dump(config))
    
    base_weights = config['base_weights'] 

    dataset_ = 'MPII' 
    max_step = 15
    train_dict = {
        'num_trees': 5, 'tree_depth': config['tree_depth'], 'num_classes': 1, 'dataset': dataset_,
        'num_batchs_update_forest': 50, 'iterations_update_forest': 20, 'exp': Exp,
        'train_txt': '/root/new/dataset/MPIIFace-pair-RT/txt/{}train-{}.txt'.format(person_id, dataset_),
        'test_txt': '/root/new/dataset/MPIIFace-pair-RT/txt/{}test-{}.txt'.format(person_id, dataset_),
        'val_txt': '/root/new/dataset/MPIIFace-pair-RT/txt/{}test-{}.txt'.format(person_id, dataset_),
        'img_dir': '/root/new/dataset/MPIIFace-pair-RT/normalized_images/',
        'pretrain_model': (base_weights, 0, dataset_, max_step),
        'lr': config['lr'], 'max_step': max_step, 'batch_size': config['batchsize'], 'pace': 0,
        'checkpoint': checkpointsPath,
        'tensorboard':'{}/output/tensorboard/'.format(Exp),
        'MAE_savepath': '{}/output/mae/'.format(Exp),
        'lr_policy': 0
    }
    checkdir(train_dict['checkpoint'])
    checkdir(train_dict['MAE_savepath'])
    picksamples = PickSamples(exp=Exp, percent=pace_percent, pace=0, alpha=alpha, ent_threshold=config['threshold'],
        diff_threshold=1000, ent_pick_per=config['ent_pick'], random_pick=False, soft=True, soft_percent=config['soft_percent'],
        train_txt0=train_dict['train_txt'],
        img_dir=train_dict['img_dir'])

    for pace in range(0, 1+len(pace_percent)): 
        print('Pace %d' % pace)
        if pace == len(pace_percent):
            left_txt, pick_txt = picksamples.pick(pace=pace, capped=config['capped'])
        else:
            left_txt, pick_txt = picksamples.pick(pace=pace, capped=False)
        print('left_txt: %s' % left_txt)
        print('pick_txt: %s' % pick_txt)
        train_dict['pace'] = pace
        train_dict['tensorboard'] = '{}/output/tensorboard/pace{}'.format(Exp, pace)
        checkdir(train_dict['tensorboard'])
        train_dict['tb_comment'] = 'pace' + str(pace)
        if pace == 0:
            train_dict['max_step'] = 10
            train_dict['train_txt'] = left_txt
            train_dict['pretrain_model'] = (base_weights, 0, dataset_, max_step)
        elif pace == 1:
            train_dict['max_step'] = 10
            train_dict['train_txt'] = pick_txt
            train_dict['pretrain_model'] = (base_weights, 0, dataset_, max_step)
        else:
            train_dict['max_step'] = 10
            train_dict['train_txt'] = pick_txt
            train_dict['pretrain_model'] = (checkpointsPath, pace-1, dataset_, 80)
        if pace == len(pace_percent):
            train_dict['max_step'] = 10
        print(train_dict)
        # train
        train.train(train_dict)

        # predict on train set to pick samples for each pace
        para_dict = {}
        para_dict['pretrain_model'] = (checkpointsPath, pace, dataset_, 80)
        para_dict['train_txt'] = train_dict['train_txt']
        para_dict['img_dir'] = train_dict['img_dir']
        para_dict['lr'] = train_dict['lr']
        para_dict['num_trees'] = train_dict['num_trees']
        para_dict['tree_depth'] = train_dict['tree_depth']
        para_dict['pace'] = pace
        para_dict['num_classes'] = train_dict['num_classes']

        # predict on pick set
        para_dict['test_txt'] = '{}/images/Pick-{}.txt'.format(Exp, pace)
        para_dict['predict_txt'] = '{}/Pred/PredOnPickset-{}.txt'.format(Exp, pace)
        predict.Predict(para_dict)

        # predict on left set
        para_dict['test_txt'] = '{}/images/Left-{}.txt'.format(Exp, pace)
        para_dict['predict_txt'] = '{}/Pred/PredOnLeftset-{}.txt'.format(Exp, pace)
        predict.Predict(para_dict)

        # predict on test set
        para_dict['test_txt'] = train_dict['test_txt']
        para_dict['predict_txt'] = '{}/Pred/PredOnTestSet-{}.txt'.format(Exp, pace)
        predict.Predict(para_dict)

if __name__ == '__main__':

    PID = PPID  
    for i in PID:
        print('===============================\nPerson: {}'.format(i))
        step(i)