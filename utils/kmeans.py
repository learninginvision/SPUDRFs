
from sklearn.cluster import KMeans
import operator
from functools import reduce
import numpy as np 

def kmeans(x, n_kluster=5):
    km = KMeans(n_clusters=n_kluster)
    km.fit(x)
    sigma = np.zeros([n_kluster, 1])

    for idx, label in enumerate(km.labels_):
        sigma[label] += (x[idx] - km.cluster_centers_[label]) ** 2

    for i in range(n_kluster):
        sigma[i] /= len(km.labels_[km.labels_==i])

    return np.reshape(km.cluster_centers_, [-1]), np.reshape(sigma, [-1])

if __name__ == "__main__":
    train_txt = './caffe_txt1/MORPH-train.txt'
    labels = []
    with open(train_txt,"r") as f: 
        lines = f.readlines()      
        for line in lines:
            if 'noise' not in line:
                label = line.strip('\n').split(' ')[1]
                label = float(label)
                labels.append(label)
        labels = np.reshape(np.array(labels), [-1, 1])