from continuum.datasets import  ImageNet100
from continuum import ClassIncremental
from tqdm import tqdm
import torch
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize,Resize,CenterCrop
import timm
from torch.utils.data import DataLoader
from torch.nn import functional as F
import os
train_ds = ImageNet100(data_path="/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg/", train=True, download=True,
    data_subset="/ibex/ai/home/jansonrp/workshop/data/imagenet100_splits/train_100.txt")
test_ds = ImageNet100(data_path="/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg/", train=False, download=True,
    data_subset="/ibex/ai/home/jansonrp/workshop/data/imagenet100_splits/val_100.txt")

scenario_train = ClassIncremental(train_ds, increment=10,initial_increment=10,transformations=[Resize(256),CenterCrop(224),ToTensor()],
    class_order=[68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
)
scenario_test = ClassIncremental(test_ds,increment=10,initial_increment=10,transformations=[Resize(256),CenterCrop(224),ToTensor()],
    class_order=[68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50,
            28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96,
            98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69,
            36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]
)
print(torch.cuda.is_available())

vit_b_16 = timm.create_model("vit_base_patch16_224_in21k",pretrained=True).cuda()

class_mean_set = []
accuracy_history = []
for task_id, train_dataset in enumerate(scenario_train):
    train_loader = DataLoader(train_dataset, batch_size=512)
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
    for i in range(task_id * 10, (task_id+1)*10):
        image_class_mask = (y == i)
        class_mean_set.append(np.mean(X[image_class_mask],axis=0))
    test_ds = scenario_test[:task_id+1]
    test_loader = DataLoader(test_ds, batch_size=512)
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