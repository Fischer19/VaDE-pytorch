from util import *
import argparse
import numpy as np
import scipy
from scipy import cluster
import numpy as np
import torch
from sklearn.manifold import TSNE
import scipy
from scipy.cluster.hierarchy import linkage, dendrogram
from tqdm import tqdm
from sklearn.decomposition import PCA
import sklearn
from dataloader import *
from model import *

parse=argparse.ArgumentParser(description='VaDE')
parse.add_argument('--batch_size',type=int,default=800)
parse.add_argument('--datadir',type=str,default='./data/mnist')
parse.add_argument('--nClusters',type=int,default=10)

parse.add_argument('--hid_dim',type=int,default=10)
parse.add_argument('--cuda',type=bool,default=False)


args=parse.parse_args()

#DL,_=get_mnist(args.datadir,args.batch_size)
DL,_=get_20newsgroup("tfidf_embedding.pk",batch_size = 128)

with open("tfidf_embedding.pk", "rb") as f:
    dic = pickle.load(f)
    X = dic["X"].float()
    y = dic["y"]

vade=VaDE(args)
#vade=nn.DataParallel(vade,device_ids=range(1))

vade.pre_train(DL,pre_epoch=50)

Z = linkage(y[:2000].reshape(-1,1), "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
max = compute_objective_gt(2000, rootnode, y[:2000]).numpy()

mean, _ = vade.encoder(X)
mean = mean.detach().numpy()
Z = linkage(mean[:2000], "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
print("VaDE:", compute_objective_gt(2000, rootnode, y[:2000]).numpy() / max)



pca = PCA(n_components = 10)
pca_data = pca.fit_transform(X[:2000])
origin_data = X[:2000]

Z = linkage(y[:2000].reshape(-1,1), "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
max = compute_objective_gt(2000, rootnode, y[:2000]).numpy()

Z = linkage(pca_data, "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
print("PCA:", compute_objective_gt(2000, rootnode, y[:2000]).numpy() / max)


Z = linkage(y[:2000].reshape(-1,1), "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
max = compute_objective_gt(2000, rootnode, y[:2000]).numpy()

Z = linkage(origin_data, "ward")
rootnode, nodelist = scipy.cluster.hierarchy.to_tree(Z, rd=True)
print("origin:", compute_objective_gt(2000, rootnode, y[:2000]).numpy() / max)


