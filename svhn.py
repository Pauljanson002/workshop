import torch
from torchvision.datasets import SVHN
import torchvision
import torchvision.transforms as transforms
import timm
from torch.nn import functional as F
import numpy as np
svhn_train_ds = SVHN(root='./data', split='train', download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))

svhn_test_ds = SVHN(root='./data', split='test', download=True, transform=torchvision.transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))

train_loader = torch.utils.data.DataLoader(svhn_train_ds, batch_size=1024, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(svhn_test_ds, batch_size=1024, shuffle=False, num_workers=4)

vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()

class_mean_set = []
correct, total = 0, 0
X = []
y = []
for (img_batch,label) in train_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    X.append(out)
    y.append(label)
X = np.concatenate(X)
y = np.concatenate(y)
for i in range(0,10):
    image_class_mask = (y == i)
    class_mean_set.append(np.mean(X[image_class_mask],axis=0))
for (img_batch,label) in test_loader:
    img_batch = img_batch.cuda()
    with torch.no_grad():
        out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
    predictions = []
    for single_image in out:
        distance = single_image - class_mean_set
        norm = np.linalg.norm(distance,ord=2,axis=1)
        pred = np.argmin(norm)
        predictions.append(pred)
    predictions = torch.tensor(predictions)
    #predictions = predictions - 20
    correct += (predictions.cpu() == label.cpu()).sum()
    total += label.shape[0]
print(correct)
print(total)
print(correct/total)