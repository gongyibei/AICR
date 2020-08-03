# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from src.midi import *
from src.model import *
from dataset import MidiDataSet
from torch.utils.data import DataLoader


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = resnet()
model.load_state_dict(torch.load('./models/resnet2/epoch_26.pth'))
model.to(device)

dataset = MidiDataSet()
data_loader = DataLoader(dataset, batch_size=100, shuffle=True, drop_last=True)

X = []
cnt = 0 
for batch_x, _  in data_loader:
    print(batch_x.device)
    cnt += batch_x.shape[0]
    print(cnt)
    for child in list(model.children())[:-1]:
        batch_x.to(device)
        print(batch_x.device)
        batch_x = child(batch_x)
    X.append(batch_x.detach().numpy)
    
X = np.vstack(X)
X = X.reshape(X.shape[0], -1)

tsne = manifold.TSNE()
Y = tsne.fit_transform(X)
print(Y.shape)
plt.scatter(Y[:n, 0], Y[:n, 1])
plt.scatter(Y[n:, 0], Y[n:, 1])
plt.show()

