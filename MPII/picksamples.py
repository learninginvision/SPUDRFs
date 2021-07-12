import os
import numpy as np 
import random

class PickSamples():

    def __init__(self, exp=0, percent=[0.5, 0.125, 0.125, 0.125, 0.125], pace=0, alpha=[15, 12.5, 10, 7.5, 5], 
                ent_threshold=-4.2, diff_threshold=100, ent_pick_per=1200, random_pick=False,
                train_txt0='./images/tr_10.txt', soft=False, soft_percent=0.9, 
                img_dir='/root/data/aishijie/Project/Morph_mtcnn_1.3_0.35_0.3/'):
        self.exp = exp
        self.root_image = '{}/images/'.format(exp)
        self.root_Pred = '{}/Pred/'.format(exp)
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
            self.ent_pick_per = ent_pick_per

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

    # 'img name: {}, label: {}, pred: {:.6f}, ent: {:.6f}, diff: {:.6f}, head: '
    def get_img_name(self, line):
        img = line.strip('\n').split('img name: ')[1].split(',')[0]
        return img

    def get_label(self, line):
        label = line.strip('\n').split('label: ')[1].split(',')[0]
        pitch = label.split('y')[0][1:]
        yaw = label.split('y')[1]
        return float(pitch), float(yaw)

    def get_diff(self, line):
        diff = line.strip('\n').split('diff: ')[1].split(',')[0]
        return float(diff)

    def get_ent(self, line):
        ent = line.strip('\n').split('ent: ')[1].split(',')[0]
        return float(ent)

    def get_headpose(self, line):
        headpose = line.strip('\n').split('headpose: ')[-1]
        return headpose

    def pick(self, pace=0, capped=False):
        '''
        pace represent the txt need to be generated
        '''
        pick, left, pick_ent, pick_new = [],[],[],[]
        if pace == 0:
            pick = []
            left = self.readtxt(self.fn_traintxt0)
        else:
            fn_train_previous = '{}/images/Pick-{}.txt'.format(self.exp, pace-1) 
            fn_pred_pick = '{}/Pred/PredOnPickset-{}.txt'.format(self.exp, pace-1)
            fn_pred_left = '{}/Pred/PredOnLeftset-{}.txt'.format(self.exp, pace-1)
            pred_pick = self.readtxt(fn_pred_pick)
            pred_left = self.readtxt(fn_pred_left)
            pred_all = pred_pick + pred_left
            pred_pick_new = pred_pick

            # sort left samples according to diff and entopy
            pred_pick_sort = []
            for i, line in enumerate(pred_left):
                diff = self.get_diff(line)
                if diff > self.diff_threshold:
                    diff = self.diff_threshold
                img = self.get_img_name(line)
                ent = self.get_ent(line)
                label = self.get_label(line)
                headpose = self.get_headpose(line)
                if self.ent_threshold < 0:
                    if ent < self.ent_threshold:
                        ent = self.ent_threshold
                diff = diff - self.alpha[pace-1] * ent
                pred_pick_sort.append((img, label, diff, headpose))
                
            idx_pred_sort_left = [index for index,value in sorted(list(enumerate(pred_pick_sort)),key=lambda x:x[1][2])]
            
            # pick samples according to diff and entopy
            for i in range(len(pred_pick_sort)):
                idx = idx_pred_sort_left[i]
                img_name, label, headpose = pred_pick_sort[idx][0], pred_pick_sort[idx][1], pred_pick_sort[idx][3]
                line = '{} {} {} {} {}\n'.format(img_name, label[0], label[1], 10000, headpose)
                if i < self.pace_samples[pace-1]:
                    pick.append(line)
                    pred_pick_new.append(pred_left[idx])
                else:
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

                    previous_ent = []
                    for p in range(pace):
                        fn_ent_pick = '{}/images/ent_pick-{}.txt'.format(self.exp, p)
                        previous_ent = previous_ent + self.readtxt(fn_ent_pick)
                    cnt = 0
                    for i in range(idx_ent.shape[0]):
                        if cnt >= self.ent_pick_per:
                            break
                        idx = idx_ent[i]
                        line_ = pred_all[idx]
                        img = self.get_img_name(line_)

                        # 查询是否已经增强
                        flag = True
                        for prev in previous_ent:
                            info = prev.strip('\n').split(' ')
                            img_prev = '{} {}'.format(info[0], info[1])
                            if img == img_prev:
                                flag = False
                                break
                        # 增强
                        if flag:
                            label = self.get_label(line_)
                            headpose = self.get_headpose(line_)
                            line = '{} {} {} {} {}\n'.format(img, label[0], label[1], 10000, headpose)
                            pick_ent.append(line)
                            pred_pick_new.append(pred_all[idx])
                            cnt  = cnt + 1
            
            # Mixture Weighting
            tem = self.readtxt(fn_train_previous)
            pick_new = pick + tem + pick_ent

            if self.soft:
                img_all, pick_new_sort = [], []

                # capped likelihood
                if capped != False:
                    pred_pick_new.sort(key=lambda x:self.get_diff(x))
                    end = int(len(pred_pick_new)*capped)
                    pred_pick_new = pred_pick_new[:end]
                    pick_new = pick_new[:end]
                
                for pred in pred_pick_new:
                    diff = self.get_diff(pred)
                    img_name = self.get_img_name(pred)
                    ent = self.get_ent(pred)
                    label = self.get_label(pred)
                    headpose = self.get_headpose(pred)
                    if self.ent_threshold < 0:
                        if ent < self.ent_threshold:
                            ent = self.ent_threshold
                    diff = diff - self.alpha[pace-1] * ent
                    pick_new_sort.append((img_name, label, diff, headpose))
                    
                pick_new_sort.sort(key=lambda x:x[2])
                num_pick = len(pick_new_sort)
                
                # linear weighting
                # lambda0 = pick_new_sort[-1][2]
                # for i, (img, label, diff, headpose) in enumerate(pick_new_sort):
                #     weight = 10000.0 * (lambda0 - diff) / lambda0
                #     pick_new[i] = '{} {} {} {} {}\n'.format(img, label[0], label[1], weight, headpose)

                # log weighting
                # max_val, min_val = np.max(diffs), np.min(diffs)
                # interval = max_val - min_val
                # lambda0 = ((pick_new_sort[-1][2]-min_val) / interval) * 0.8 + 0.1
                # print(lambda0)
                # for i, (img, label, diff, headpose) in enumerate(pick_new_sort):
                #     diff = ( (diff-min_val) / interval ) * 0.8 + 0.1
                #     weight = 10000.0 * 1.0 / np.log(1-lambda0) * np.log(diff+1-lambda0)
                #     pick_new[i] = '{} {} {} {} {}\n'.format(img, label[0], label[1], weight, headpose)

                lambda_0 = pick_new_sort[-1][2] # 12
                lambda_1 = pick_new_sort[int(num_pick*self.soft_percent-2)][2] # 4
                tmp = 1/lambda_1 - 1/lambda_0
                epsilon = 0.0
                if abs(tmp) < 1e-5:
                    epsilon = 0.0
                else :
                    epsilon = 1 / (tmp)
                print('lambda_0: {}, lambda_1: {}, epsilon: {}'.format(lambda_0, lambda_1, epsilon))
                weight = 0
                for i, (img, label, diff, headpose) in enumerate(pick_new_sort):
                    if i < num_pick*self.soft_percent:
                        weight = 10000
                    else:
                        weight = int(10000*(epsilon / diff - epsilon / lambda_0))
                    pick_new[i] = '{} {} {} {} {}\n'.format(img, label[0], label[1], weight, headpose)
            
            
        # save txt
        fn_pick_new = '{}/images/Pick-{}.txt'.format(self.exp, pace)
        fn_left_new = '{}/images/Left-{}.txt'.format(self.exp, pace)
        fn_pick_ent = '{}/images/ent_pick-{}.txt'.format(self.exp, pace)
        self.savetxt(fn_pick_new, pick_new)
        self.savetxt(fn_left_new, left)
        self.savetxt(fn_pick_ent, pick_ent)

        print('new pick: %d' % len(pick_new))
        print('entropy pick: %d' % len(pick_ent))
        print('new left: %d' % len(left))
        return (fn_left_new, fn_pick_new)

