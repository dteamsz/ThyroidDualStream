from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from imgaug import augmenters as iaa
import random
import timm
import cv2
import seaborn as sns
from eval_tool import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--mode")
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()
print('current mode is {}'.format(args.mode))
print('current checkpoint is {}'.format(args.checkpoint))

import torch.nn.functional as F
def read_avi(path, force2gray=True):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if force2gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    frames = np.array(frames)
    return frames

def delay_pad_avi(vid, start, Sample_Bch):
    lenVid = len(vid)
    idx = np.linspace(start,lenVid-1,Sample_Bch).tolist()
    idx = [int(x) for x in idx]
    vid = vid[idx]
    return vid

class Feature_Extractor(nn.Module):
    def __init__(self, model, mode='feature_extraction'):
        super(Feature_Extractor, self).__init__()
        # 取掉model的后1层
        self.model_layer = nn.Sequential(*list(model.children())[:-2])
    def forward(self, x):
        x = self.model_layer(x)
      # Perform the usual forward pass
        return x

class TotalModel(nn.Module):
    def __init__(self, img_feature_extractor, roi_feature_extractor=None, mode='dual_lstm'):
        super(TotalModel, self ).__init__()
        # 取掉model的后1层
        self.img_feature_extractor = img_feature_extractor
        self.roi_feature_extractor = roi_feature_extractor
        self.mode = mode
        if mode == 'dual_maxpool':
            self.Maxpool = nn.AdaptiveMaxPool2d((1,3072))
            self.dual_maxpool_fc = nn.Linear(3072,2)
        elif mode == 'dual_averagepool':
            self.Averagepool = nn.AdaptiveAvgPool2d((1,3072))
            self.dual_averagepool_fc = nn.Linear(3072,2)
        elif mode == 'single_maxpool':
            self.Maxpool = nn.AdaptiveMaxPool2d((1,1536))
            self.single_maxpool_fc = nn.Linear(1536,2)
        elif mode == 'single_averagepool':
            self.Averagepool = nn.AdaptiveAvgPool2d((1,1536))
            self.single_averagepool_fc = nn.Linear(1536,2)
        elif mode == 'dual_T_conv':
            self.T_conv = nn.Conv2d(1,1,(64,1),1)
            self.dual_T_conv_fc = nn.Linear(3072,2) 
        elif self.mode == 'single_T_conv':
            self.T_conv = nn.Conv2d(1,1,(64,1),1)
            self.single_T_conv_fc = nn.Linear(1536,2) 

    def forward(self, x_img, x_roi):
        x_img = x_img.squeeze(0)
        x_roi = x_roi.squeeze(0)
        img_feature = self.img_feature_extractor(x_img)
        if 'dual' in self.mode:
            roi_feature = self.roi_feature_extractor(x_roi)
            feature = torch.cat((img_feature,roi_feature),1).unsqueeze(0)
            if self.mode == 'dual_maxpool': 
                result = self.dual_maxpool_fc(self.Maxpool(feature).squeeze(0))
            elif self.mode == 'dual_averagepool':
                result = self.dual_averagepool_fc(self.Averagepool(feature).squeeze(0))
            elif self.mode == 'dual_T_conv':
                result = self.dual_T_conv_fc(self.T_conv(feature).squeeze(0))
        elif 'single' in self.mode:
            if self.mode == 'single_maxpool': 
                result = self.single_maxpool_fc(self.Maxpool(img_feature.unsqueeze(0)).squeeze(0))
            elif self.mode == 'single_averagepool':
                result = self.single_averagepool_fc(self.Averagepool(img_feature.unsqueeze(0)).squeeze(0))
            elif self.mode == 'single_T_conv':
                x = self.T_conv(img_feature.unsqueeze(0)).squeeze(0)
                result = self.single_T_conv_fc(x)
                
        # result = F.softmax(result, dim=1)
        return result

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, usage='train'):
        self.df = pd.read_excel(annotations_file)
        self.df = self.df[self.df['data_usage']==usage]
        self.df = self.df.reset_index(drop=True)
        self.usage = usage
        self.seq = iaa.SomeOf((0,5),[
            iaa.Affine(  # 对一部图像做仿射变换
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # 平移±20%之间
                rotate=(-15, 15),  # 旋转±30度之间
                order=[0, 1],  # 使用最邻近差值或者双线性差值
                cval=(0),  # 全白全黑填充"constant"
                mode="constant"  # mode=ia.ALL    #定义填充图像外区域的方法
            ),
            iaa.Fliplr(0.5),
            iaa.Dropout2d(p=0.5),
            iaa.GaussianBlur(sigma=(0, 2.0), name=None,  deterministic="deprecated", random_state=None),
            iaa.AverageBlur(k=(1, 3), name=None,  deterministic="deprecated", random_state=None),
            iaa.MultiplyElementwise(mul=(0.5, 1.5), per_channel=False, name=None,  deterministic="deprecated", random_state=None),
            iaa.imgcorruptlike.SpeckleNoise(severity=2),
            ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'ROI'].replace('./','/hy-tmp/')
        roi_path = img_path.replace('.avi','_OD.avi')
        random.seed(42)
        st = random.randint(0,3)
        imgs = read_avi(img_path)
        imgs = delay_pad_avi(imgs, st, 64)
        rois = read_avi(roi_path)
        rois = delay_pad_avi(rois, st, 64)
        
        if self.usage == 'train':
            if random.random() < 0.5:
                imgs = self.seq.augment_images(imgs)
                rois = self.seq.augment_images(rois)
        new_imgs = []
        for i in range(len(imgs)):
            new_imgs.append(cv2.resize(imgs[i],(256,256)))
        new_imgs = np.array(new_imgs)
        imgs = torch.from_numpy(new_imgs).float()/255
        imgs = imgs.unsqueeze(1)
        rois = torch.from_numpy(rois).float()/255
        rois = rois.unsqueeze(1)
        labels = torch.tensor(self.df.loc[idx, 'malignancy']-1).long()
        weight = len(self.df)/len(self.df[self.df['malignancy']==self.df.loc[idx, 'malignancy']])
        weight = torch.tensor(weight)
        return imgs, rois, labels, weight

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), './ckpt/best_checkpoint_{}.pt'.format(args.mode))
        self.val_loss_min = val_loss

def train():
    """
    LOAD MODEL
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    feature_extractor = Feature_Extractor(timm.create_model('inception_resnet_v2', in_chans=1, num_classes=2))
    state_dict = torch.load('./ckpt/feature_extraction_inception-resnetv2.pt', map_location='cuda:0')
    feature_extractor.load_state_dict(state_dict)
    model = TotalModel(feature_extractor, feature_extractor, mode=args.mode)
    model = model.to(device)

    """
    Freeze BatchNorm
    Because batchsize equals to 1
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

    """
    PREPARE DATA
    """
    Catadf = pd.read_excel("./Thyroid_Dataset.xlsx")
    Catadf = Catadf[Catadf['视频类型（干净单个1、不干净单个2、全景3）'].isin([1,2])]
    Catadf.reset_index(drop=True, inplace=True)

    data_train = CustomImageDataset(annotations_file= './Thyroid_Dataset.xlsx', usage='train')
    data_valid = CustomImageDataset(annotations_file= './Thyroid_Dataset.xlsx', usage='valid')
    batch_size = 1

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=1,
                                                    pin_memory=True)

    data_loader_valid = torch.utils.data.DataLoader(dataset=data_valid,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=1,
                                                    pin_memory=True)

    # # 实例化模型并传输到设备
    epochs = 100
    loss_fn = nn.CrossEntropyLoss()
    base_lr = 1e-3
    optimizer = optim.SGD([
                {'params': model.parameters()}],lr=base_lr, weight_decay=1e-3)
    best_loss = 1e10
    early_stopping = EarlyStopping(patience=3, verbose=True)

    best_acc = 0
    for epoch in range(epochs):
        print('**************epoch:%d**************' % epoch)
        sum_loss=0
        cm_caches = np.zeros((2,2))
        model.train()
        """
        Freeze BatchNorm
        Because batchsize equals to 1
        """
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
        for data in tqdm(data_loader_train):
            
            imgs, rois, labels, weight = data
            imgs, rois, labels, weight = imgs.to(device), rois.to(device), labels.to(device), weight.to(device)
            outs = model(imgs, rois)
            loss = loss_fn(outs, labels)*weight
            # print(loss)
            sum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            label_idx = labels.detach().cpu().numpy()
            pred_idx = outs[0].detach().cpu().numpy()
            pred_idx = np.argmax(pred_idx)
            cm_caches[label_idx,pred_idx] += 1
        fig = plt.figure(figsize=(10,10))
        sns.heatmap(cm_caches, annot=True, fmt='.20g', cmap='Blues')
        fig.savefig('./confusion_matrix_train.png')
        print('acc:',np.sum(np.diag(cm_caches))/np.sum(cm_caches))
        print("train_loss:%.03f" % (sum_loss / len(data_loader_train)))

        sum_loss=0
        cm_caches = np.zeros((2,2))    
        model.eval()
        with torch.no_grad():
            for data in tqdm(data_loader_valid):
                imgs, rois, labels, weight = data
                imgs, rois, labels, weight = imgs.to(device), rois.to(device), labels.to(device), weight.to(device)
                outs = model(imgs, rois)
                loss = loss_fn(outs, labels)*weight
                sum_loss += loss.item()

                label_idx = labels.detach().cpu().numpy()
                pred_idx = outs[0].detach().cpu().numpy()
                pred_idx = np.argmax(pred_idx)
                cm_caches[label_idx,pred_idx] += 1
        
        fig = plt.figure(figsize=(10,10))
        sns.heatmap(cm_caches, annot=True, fmt='.20g', cmap='Blues')
        fig.savefig('./confusion_matrix_val.png')
        print('acc:',np.sum(np.diag(cm_caches))/np.sum(cm_caches))
        print("valid_loss:%.03f" % (sum_loss / len(data_loader_valid)))
        torch.save(model.state_dict(), './pth_log/{}_{}_{:.3f}.pt'.format(args.mode,epoch,(sum_loss / len(data_loader_valid))))
        early_stopping(sum_loss/len(data_loader_valid), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def test(checkpoint):
    """
    LOAD MODEL
    """

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    feature_extractor = Feature_Extractor(timm.create_model('inception_resnet_v2', in_chans=1, num_classes=2))
    state_dict = torch.load('./ckpt/feature_extraction_inception-resnetv2.pt', map_location='cuda:0')
    feature_extractor.load_state_dict(state_dict)
    model = TotalModel(feature_extractor, feature_extractor, mode=args.mode)
    if checkpoint is None:
        model.load_state_dict(torch.load('./ckpt/best_checkpoint_{}.pt'.format(args.mode), map_location='cuda:0'))
    else:
        model.load_state_dict(torch.load(checkpoint,map_location='cuda:0'))
    model = model.to(device)


    """
    PREPARE DATA
    """
    Catadf = pd.read_excel("./Thyroid_Dataset.xlsx")
    Catadf = Catadf[Catadf['视频类型（干净单个1、不干净单个2、全景3）'].isin([1,2])]
    Catadf.reset_index(drop=True, inplace=True)

    df = Catadf[Catadf['data_usage']=='test']
    df.reset_index(drop=True, inplace=True)

    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(df))):
            
            img_path = df.loc[i, 'ROI']
            roi_path = img_path.replace('.avi','_OD.avi')
            st = 0
            imgs = read_avi(img_path)
            imgs = delay_pad_avi(imgs, st, 64)
            rois = read_avi(roi_path)
            rois = delay_pad_avi(rois, st, 64)
            
            new_imgs = []
            for j in range(len(imgs)):
                new_imgs.append(cv2.resize(imgs[j],(256,256)))
            new_imgs = np.asarray(new_imgs)
            imgs = torch.from_numpy(new_imgs).float()/255
            imgs = imgs.unsqueeze(1).unsqueeze(0)
            rois = torch.from_numpy(rois).float()/255
            rois = rois.unsqueeze(1).unsqueeze(0)
            imgs, rois = imgs.to(device), rois.to(device)
            outs = model(imgs, rois)
            outs = nn.Softmax(dim=1)(outs)
            pred = outs[0,1].detach().cpu().numpy()
            y_pred.append(pred)
            y_true.append(df.loc[i, 'malignancy']-1)


    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout(pad=2, w_pad=2.)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    thresh = get_auc(axes[0,0], np.array(y_true), np.array(y_pred), title='ROC')
    print(thresh)
    get_precision_recall(axes[0, 1], np.array(y_true), np.array(y_pred), title='PR')
    y_pred_lvl = [1 if y>thresh else 0 for y in y_pred]
    cm = confusion_matrix(y_true,y_pred_lvl)
    plot_confusion_matrix(axes[1,0], cm,target_names=['0','1'],normalize=False)
    plot_confusion_matrix(axes[1,1], cm,target_names=['0','1'],normalize=True)
    fig.figure.savefig('./Video_Level Test {}.png'.format(args.mode))
    df['y_pred'] = y_pred
    df.to_excel('./test_result_{}.xlsx'.format(args.mode), index=False, header=True)
    IDs = df['StudyID'].unique()
    y_true = []
    y_pred = []
    for ID in IDs:
        df_temp = df[df['StudyID'] == ID]
        df_temp.reset_index(drop=True, inplace=True)
        y_true.append(df_temp['malignancy'][0]-1)
        y_pred.append(np.mean(df_temp['y_pred']))
    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout(pad=2, w_pad=2.)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    get_auc(axes[0,0], np.array(y_true), np.array(y_pred), title='ROC')
    get_precision_recall(axes[0, 1], np.array(y_true), np.array(y_pred), title='PR')
    y_pred_lvl = [1 if y>thresh else 0 for y in y_pred]
    cm = confusion_matrix(y_true,y_pred_lvl)
    plot_confusion_matrix(axes[1,0], cm,target_names=['0','1'],normalize=False)
    plot_confusion_matrix(axes[1,1], cm,target_names=['0','1'],normalize=True)
    fig.figure.savefig('./Patient_Level Test {}.png'.format(args.mode))

    df_ = df.drop_duplicates(subset=['StudyID'])
    df_.reset_index(drop=True, inplace=True)
    df_['ODRaw'] = y_pred
    df_.to_excel('./patient_result_{}.xlsx'.format(args.mode), index=False, header=True)

def external_test(checkpoint):
    """
    LOAD MODEL
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    feature_extractor = Feature_Extractor(timm.create_model('inception_resnet_v2', in_chans=1, num_classes=2))
    state_dict = torch.load('./ckpt/feature_extraction_inception-resnetv2.pt', map_location='cuda:0')
    feature_extractor.load_state_dict(state_dict)
    model = TotalModel(feature_extractor, feature_extractor, mode=args.mode)
    if checkpoint is None:
        model.load_state_dict(torch.load('./ckpt/best_checkpoint_{}.pt'.format(args.mode), map_location='cuda:0'))
    else:
        model.load_state_dict(torch.load(checkpoint, map_location='cuda:0'))
    model = model.to(device)


    """
    PREPARE DATA
    """
    df = pd.read_excel("./external_test.xlsx")


    y_pred = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(df))):
            
            img_path = df.loc[i, 'ROI'].replace('./','/hy-tmp/')
            roi_path = img_path.replace('.avi','_OD.avi')
            st = 0
            imgs = read_avi(img_path)
            imgs = delay_pad_avi(imgs, st, 64)
            rois = read_avi(roi_path)
            rois = delay_pad_avi(rois, st, 64)
            
            # if self.usage == 'train':
            #     imgs = self.seq.augment_images(imgs)
            #     rois = self.seq.augment_images(rois)
            new_imgs = []
            for j in range(len(imgs)):
                new_imgs.append(cv2.resize(imgs[j],(256,256)))
            new_imgs = np.asarray(new_imgs)
            imgs = torch.from_numpy(new_imgs).float()/255
            imgs = imgs.unsqueeze(1).unsqueeze(0)
            rois = torch.from_numpy(rois).float()/255
            rois = rois.unsqueeze(1).unsqueeze(0)
            imgs, rois = imgs.to(device), rois.to(device)
            outs = model(imgs, rois)
            outs = nn.Softmax(dim=1)(outs)
            pred = outs[0,1].detach().cpu().numpy()
            y_pred.append(pred)
            y_true.append(df.loc[i, 'malignancy'])


    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.tight_layout(pad=2, w_pad=2.)
    fig.set_figheight(8)
    fig.set_figwidth(10)

    thresh = get_auc(axes[0,0], np.array(y_true), np.array(y_pred), title='ROC')
    print(thresh)
    # thresh = 0.5
    get_precision_recall(axes[0, 1], np.array(y_true), np.array(y_pred), title='PR')
    y_pred_lvl = [1 if y>thresh else 0 for y in y_pred]
    cm = confusion_matrix(y_true,y_pred_lvl)
    plot_confusion_matrix(axes[1,0], cm,target_names=['0','1'],normalize=False)
    plot_confusion_matrix(axes[1,1], cm,target_names=['0','1'],normalize=True)
    fig.figure.savefig('./Video_Level External Test {}.png'.format(args.mode))
    df['y_pred'] = y_pred
    df.to_excel('./external_test_result_{}.xlsx'.format(args.mode), index=False, header=True)

if __name__ == '__main__':
    # train()
    test(args.checkpoint)
    # external_test(args.checkpoint)
