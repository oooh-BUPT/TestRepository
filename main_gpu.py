 import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as tfs
from torchvision import models
from torch import nn
 
import matplotlib.pyplot as plt
 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

use_gpu = True
train_transform = tfs.Compose([
    # 训练集的数据预处理
    tfs.Resize([224, 224]),
    tfs.ToTensor(),
    tfs.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])
 
test_transform = tfs.Compose([
    tfs.Resize([224,224]),
    tfs.ToTensor(),
    tfs.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])
 
# 每一个batch的数据集数目
batch_size = 10
# 构建训练集和验证集
# 
train_set = ImageFolder('./dataset1/train', train_transform)
train_data = DataLoader(train_set, batch_size, shuffle=True, num_workers=0)
 
valid_set = ImageFolder('./dataset1/valid', test_transform)
valid_data = DataLoader(valid_set, batch_size, shuffle=False, num_workers=0)
 
train_set.class_to_idx
 
len(valid_data)
 
def get_model():
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 30)
    return model
 
model = get_model()
with torch.no_grad():
    scorce = model(iter(train_data).next()[0])
    print(scorce.shape[0], scorce.shape[1])
if use_gpu:
    model = model.cuda()
# 构建loss函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
 
# 训练的epoches数目
max_epoch = 15

def train(model, train_data, valid_data, max_epoch, criterion, optimizer):
    freq_print = int(len(train_data) / 3)
    
    metric_log = dict()
    metric_log['train_loss'] = list()
    metric_log['train_acc'] = list()
    if valid_data is not None:
        metric_log['valid_loss'] = list()
        metric_log['valid_acc'] = list()
    
    for e in range(max_epoch):
        model.train()
        running_loss = 0
        running_acc = 0
 
        for i, data in enumerate(train_data, 1):
            img, label = data
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
 
            # forward前向传播
            out = model(img)
 
            # 计算误差
            loss = criterion(out, label.long())
 
            # 反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
 
            # 计算准确率
            _, pred = out.max(1)
            num_correct = (pred == label.long()).sum().item()
            acc = num_correct/img.shape[0]
 
            running_loss += loss.item()
            running_acc +=acc
 
            if i % freq_print == 0:
                print('[{}]/[{}], train loss: {:.3f}, train acc: {:.3f}' \
                .format(i, len(train_data), running_loss / i, running_acc / i))
        
        metric_log['train_loss'].append(running_loss / len(train_data))
        metric_log['train_acc'].append(running_acc / len(train_data))
 
        if valid_data is not None:
            model.eval()
            running_loss = 0
            running_acc = 0
            for data in valid_data:
                img, label = data
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                
                # forward前向传播
                out = model(img)
 
                # 计算误差
                loss = criterion(out, label.long())
 
                # 计算准确度
                _, pred = out.max(1)
                num_correct = (pred==label.long()).sum().item()
                acc = num_correct/img.shape[0]
 
 
                running_loss += loss.item()
                running_acc += acc
 
            metric_log['valid_loss'].append(running_loss/len(valid_data))
            metric_log['valid_acc'].append(running_acc/len(valid_data))
            print_str = 'epoch: {}, train loss: {:.3f}, train acc: {:.3f}, \
            valid loss: {:.3f}, valid accuracy: {:.3f}'.format(
                        e+1, metric_log['train_loss'][-1], metric_log['train_acc'][-1],
                        metric_log['valid_loss'][-1], metric_log['valid_acc'][-1])
        else:
            print_str = 'epoch: {}, train loss: {:.3f}, train acc: {:.3f}'.format(
                e+1,
                metric_log['train_loss'][-1],
                metric_log['train_acc'][-1])
        print(print_str)
 
        
    # 可视化
    nrows = 1
    ncols = 2
    figsize= (10, 5)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    if valid_data is not None:
        figs[0].plot(metric_log['train_loss'], label='train loss')
        figs[0].plot(metric_log['valid_loss'], label='valid loss')
        figs[0].axes.set_xlabel('loss')
        figs[0].legend(loc='best')
        figs[1].plot(metric_log['train_acc'], label='train acc')
        figs[1].plot(metric_log['valid_acc'], label='valid acc')
        figs[1].axes.set_xlabel('acc')
        figs[1].legend(loc='best')
    else:
        figs[0].plot(metric_log['train_loss'], label='train loss')
        figs[0].axes.set_xlabel('loss')
        figs[0].legend(loc='best')
        figs[1].plot(metric_log['train_acc'], label='train acc')
        figs[1].axes.set_xlabel('acc')
        figs[1].legend(loc='best')
# 用作调参
train(model, train_data, valid_data, max_epoch, criterion, optimizer)

# 保存模型
torch.save(model.state_dict(), './model/save_model2.pth')