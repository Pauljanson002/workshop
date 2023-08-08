from continuum.datasets import  ImageNet1000
from continuum import ClassIncremental
from tqdm import tqdm
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize,Resize,CenterCrop
import timm
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
import json
train_ds = ImageNet1000(data_path="/ibex/ai/reference/CV/ILSVR/2021/data/resized", train=True, download=True,)
test_ds = ImageNet1000(data_path="/ibex/ai/reference/CV/ILSVR/2021/data/resized", train=False, download=True,)

scenario_train = ClassIncremental(train_ds, increment=100,initial_increment=100,transformations=[Resize(256),CenterCrop(224),ToTensor()])
scenario_test = ClassIncremental(test_ds,increment=100,initial_increment=100,transformations=[Resize(256),CenterCrop(224),ToTensor()])
print(torch.cuda.is_available())

vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()

class_mean_set = []
accuracy_history = []
for task_id, train_dataset in enumerate(scenario_train):
    train_loader = DataLoader(train_dataset, batch_size=2048)
    X = []
    y = []
    for (img_batch,label,t) in tqdm(train_loader,desc=f"Training task {task_id}",total=len(train_loader)):
        img_batch = img_batch.cuda()
        with torch.no_grad():
            out = F.normalize(vit_b_16.forward_features(img_batch)[:,0].detach()).cpu().numpy()
        X.append(out)
        y.append(label)
    X = np.concatenate(X)
    y = np.concatenate(y)
    for i in range(task_id * 100, (task_id+1)*100):
        image_class_mask = (y == i)
        class_mean_set.append(np.mean(X[image_class_mask],axis=0))
    test_ds = scenario_test[:task_id+1]
    test_loader = DataLoader(test_ds, batch_size=2048)
    correct , total = 0 , 0
    for (img_batch,label,t) in tqdm(test_loader,desc=f"Testing task {task_id}",total=len(test_loader)):
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
        correct += (predictions.cpu() == label.cpu()).sum()
        total += label.shape[0]
    print(f"Accuracy at {task_id} {correct/total}")
    accuracy_history.append(correct/total)

print(f"average incremental accuracy {round(np.mean(np.array(accuracy_history))* 100,2)} ")