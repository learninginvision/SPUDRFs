import os
import numpy as np 
import random

class PickSamples():

    def __init__(self, exp=0, percent=[0.5, 0.125, 0.125, 0.125, 0.125], pace=0, alpha=[15, 12.5, 10, 7.5, 5], 
                ent_threshold=-4.2, diff_threshold=100, ent_pick_per=1200, random_pick=False,
                train_txt0='./images/tr_10.txt', soft=False, soft_percent=0.9,
                img_dir='/root/data/aishijie/Project/Morph_mtcnn_1.3_0.35_0.3/'):
        self.exp = exp
        self.root_image = './Exp{}/images/'.format(exp)
        self.root_Pred = './Exp{}/Pred/'.format(exp)
        self.checkdir(self.root_image)
        self.checkdir(self.root_Pred)
        
        self.percent = percent
        self.pace = pace
        self.soft = soft
        self.soft_percent = soft_percent
        self.ent_threshold = ent_threshold
        self.diff_threshold = diff_threshold
        self.alpha = alpha
        self.random_pick = random_pick
        self.img_dir = img_dir
    
        self.fn_traintxt0 =train_txt0
        train_images = self.readtxt(self.fn_traintxt0)
        self.num_imgs = len(train_images)
        self.pace_samples = [int(p*len(train_images)) for p in self.percent]

        assert ent_pick_per >= 0.0, 'Curriculum Reconstruction Samples should greater than 0'
        if ent_pick_per < 1:
            self.ent_pick_per = int(ent_pick_per * self.num_imgs)
        else:
            self.ent_pick_per = int(ent_pick_per)

    def checkdir(self, tmp_dir):
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)

    def readtxt(self, fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        return lines

    def savetxt(self, fn, lines):
        with open(fn, 'w') as f:
            f.writelines(lines)

    # 'img name: {}, label: {}, pred: {:.6f}, ent: {:.6f}, diff: {:.6f}'
    def get_img_name(self, line):
        img = line.strip('\n').split('img name: ')[1].split(',')[0]
        return img

    def get_label(self, line):
        label = line.strip('\n').split('label: ')[1].split(',')[0]
        return float(label)

    def get_diff(self, line):
        diff = line.strip('\n').split('diff: ')[-1]
        return float(diff)*100.0

    def get_ent(self, line):
        ent = line.strip('\n').split('ent: ')[1].split(',')[0]
        return float(ent)

    def pick(self, pace=0, capped=False):
        '''
        pace represent the txt need to be generated
        '''
        pick, left, pick_ent, pick_new = [],[],[],[]
        if pace == 0:
            pick = []
            left = self.readtxt(self.fn_traintxt0)
        else:
            fn_train_previous = './Exp{}/images/Pick-{}.txt'.format(self.exp, pace-1) 
            fn_pred_pick = './Exp{}/Pred/PredOnPickset-{}.txt'.format(self.exp, pace-1)
            fn_pred_left = './Exp{}/Pred/PredOnLeftset-{}.txt'.format(self.exp, pace-1)
            pred_pick = self.readtxt(fn_pred_pick)
            pred_left = self.readtxt(fn_pred_left)
            pred_all = pred_pick + pred_left

            # sort left samples according to diff and entopy
            pred_pick_sort = []
            for i, line in enumerate(pred_left):
                diff = self.get_diff(line)
                if diff > self.diff_threshold:
                    diff = self.diff_threshold
                img = self.get_img_name(line)
                ent = self.get_ent(line)
                label = self.get_label(line)
                if self.ent_threshold < 0:
                    if ent < self.ent_threshold:
                        ent = self.ent_threshold
                diff = diff - self.alpha[pace-1] * ent
                pred_pick_sort.append((img, label, diff))
            pred_pick_sort.sort(key=lambda x:x[2])

            # pick samples according to diff and entopy
            for i in range(len(pred_pick_sort)):
                img_name, label = pred_pick_sort[i][0], pred_pick_sort[i][1]
                line = img_name + ' ' + str(label) + '\n'
                if i < self.pace_samples[pace-1]:
                    line = img_name + ' ' + str(label) + '\n'
                    pick.append(line)
                else:
                    line = img_name + ' ' + str(label) + ' 10000' + '\n'
                    left.append(line)
          
            # Curriculum Reconstruction
            if self.ent_pick_per > 0:
                if self.random_pick:
                    lines = self.readtxt(self.fn_traintxt0)
                    random.shuffle(lines)
                    pick_ent = lines[:self.ent_pick_per]
                else:
                    ent_sort = []
                    for line in pred_all:
                        ent = self.get_ent(line)
                        ent_sort.append(ent)
                    ent_sort_np = np.array(ent_sort)
                    idx_ent = np.argsort(-ent_sort_np)
                    idx_ent_pick = idx_ent[:self.ent_pick_per]
                    
                    for i in range(idx_ent_pick.shape[0]):
                        idx = idx_ent_pick[i]
                        line_ = pred_all[idx]
                        img = self.get_img_name(line_)
                        label = str(self.get_label(line_))
                        line = img + ' ' + label + '\n'
                        pick_ent.append(line)
            
            # Mixture Weighting
            tem_ = self.readtxt(fn_train_previous)
            tem = []
            if self.soft:
                for t in tem_:
                    img_name, label = t.strip('\n').split(' ')[0], t.strip('\n').split(' ')[1]
                    line = img_name + ' ' + label + '\n'
                    tem.append(line)
            pick_new = pick + tem + pick_ent

            if self.soft:
                img_all, pick_new_sort, pred_pick_new = [], [], []
                
                for pred in pred_all:
                    img_name = self.get_img_name(pred)
                    img_all.append(img_name)
                for p in pick_new:
                    img_name = p.split(' ')[0]
                    idx = img_all.index(img_name)
                    pred_pick_new.append(pred_all[idx])

                # capped likelihood
                if capped != False:
                    pred_pick_new.sort(key=lambda x:self.get_diff(x))
                    end = int(len(pred_pick_new)*capped)
                    pred_pick_new = pred_pick_new[:end+1]
                    pick_new = pick_new[:end+1]
                
                diffs = []
                for pred in pred_pick_new:
                    diff = self.get_diff(pred)
                    img_name = self.get_img_name(pred)
                    ent = self.get_ent(pred)
                    label = self.get_label(pred)
                    if self.ent_threshold < 0:
                        if ent < self.ent_threshold:
                            ent = self.ent_threshold
                    diff = diff - self.alpha[pace-1] * ent
                    pick_new_sort.append((img_name, label, diff))
                    diffs.append(diff)
                pick_new_sort.sort(key=lambda x:x[2])
                num_pick = len(pick_new_sort)
                diffs.sort(key=lambda x:x)
                diffs = np.array(diffs).reshape(-1, 1)
                with open('./Exp{}/images/{}diff.txt'.format(self.exp, pace), 'w') as f4:
                    np.savetxt(f4, diffs, delimiter='\t', newline='\n')
                

                # linear weighting
                # lambda0 = pick_new_sort[-1][2]
                # for i, (img, label, diff) in enumerate(pick_new_sort):
                #     weight = 10000.0 * (lambda0 - diff) / lambda0
                #     pick_new[i] = img + ' ' + str(label) + ' ' + str(weight) + '\n'

                # log weighting
                # E9: diff /100.0
                # max_val, min_val = np.max(diffs), np.min(diffs)
                # interval = max_val - min_val
                # lambda0 = ((pick_new_sort[-1][2]-min_val) / interval) * 0.8 + 0.1
                # print(lambda0)
                # for i, (img, label, diff) in enumerate(pick_new_sort):
                #     diff = ( (diff-min_val) / interval ) * 0.8 + 0.1
                #     weight = 10000.0 * 1.0 / np.log(1-lambda0) * np.log(diff+1-lambda0)
                #     pick_new[i] = img + ' ' + str(label) + ' ' + str(weight) + '\n'


                # mixture weighting
                lambda_0 = pick_new_sort[-1][2] # 12
                lambda_1 = pick_new_sort[int(num_pick*self.soft_percent)-2][2] # 4
                tmp = 1/lambda_1 - 1/lambda_0
                epsilon = 0.0
                if abs(tmp) < 1e-5:
                    epsilon = 0.0
                else :
                    epsilon = 1 / (tmp)
                print('lambda_0: {}, lambda_1: {}, epsilon: {}'.format(lambda_0, lambda_1, epsilon))
                weight = 0
                for i, (img, label, diff) in enumerate(pick_new_sort):
                    if i < num_pick*self.soft_percent:
                        weight = 10000
                    else:
                        weight = int(10000*(epsilon / diff - epsilon / lambda_0))
                    pick_new[i] = img + ' ' + str(label) + ' ' + str(weight) + '\n'
            
        # save txt
        fn_pick_new = './Exp{}/images/Pick-{}.txt'.format(self.exp, pace)
        fn_left_new = './Exp{}/images/Left-{}.txt'.format(self.exp, pace)
        fn_pick_ent = './Exp{}/images/ent_pick-{}.txt'.format(self.exp, pace)
        self.savetxt(fn_pick_new, pick_new)
        self.savetxt(fn_left_new, left)
        self.savetxt(fn_pick_ent, pick_ent)

        print('new pick: %d' % len(pick_new))
        print('entropy pick: %d' % len(pick_ent))
        print('new left: %d' % len(left))
        return (fn_left_new, fn_pick_new)
