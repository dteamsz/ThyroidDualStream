import os
# from keras import backend as K
import pandas as pd
import numpy as np
import cv2
from skimage.transform import resize



def getallfilesofwalk(root):
    """
    使用listdir循环遍历文件夹中所有文件
    """
    if not os.path.isdir(root):
        print(root)
        return []

    dirlist = os.walk(root)
    allfiles = []
    for root, dirs, files in dirlist:
        for file in files:
            #            print(os.path.join(root, file))
            allfiles.append(os.path.join(root, file))

    return allfiles


def create_dir(dir_name):
    try:
        # Create target Directory
        os.makedirs(dir_name)
    except FileExistsError:
        print("Directory ", dir_name, " already exists")

def zero_pad(img, size=448):
    '''
    pad zeros to make a square img for resize
    '''
    h, w, c = img.shape
    if h > w:
        zeros = np.zeros([h, h - w, c]).astype(np.uint8)
        img_padded = np.hstack((img, zeros))
    elif h < w:
        zeros = np.zeros([w - h, w, c]).astype(np.uint8)
        img_padded = np.vstack((img, zeros))
    else:
        img_padded = img

    img_resized = (255*resize(img_padded, (size, size), anti_aliasing=True)).astype(np.uint8)

    return img_resized

def is_chinese(string):
    """
    检查整个字符串是否包含中文
    :param string: 需要检查的字符串
    :return: bool
    """
    for ch in string:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True

    return False


def letterbox_pad(img, size_H=256, size_W=256):
    '''
    pad zeros to make a square img for resize
    '''
    h, w, c = img.shape
    if c > 3:
        img = img[:, :, :3]
        c = 3
    if h > w:
        noise1 = np.random.uniform(.1, .3, ((h - w) // 2 * h)).reshape([h, (h - w) // 2])
        noise1 = (255 * np.concatenate((noise1[:, :, np.newaxis], noise1[:, :, np.newaxis], noise1[:, :, np.newaxis]),
                                       axis=2)).astype(np.uint8)
        noise2 = np.random.uniform(.1, .3, ((h - w) * h - (h - w) // 2 * h)).reshape([h, (h - w) - (h - w) // 2])
        noise2 = (255 * np.concatenate((noise2[:, :, np.newaxis], noise2[:, :, np.newaxis], noise2[:, :, np.newaxis]),
                                       axis=2)).astype(np.uint8)
        img_padded = np.hstack((noise1, img, noise2))
    elif h < w:
        noise1 = np.random.uniform(.1, .3, ((w - h) // 2 * w)).reshape([(w - h) // 2, w])
        noise1 = (255 * np.concatenate((noise1[:, :, np.newaxis], noise1[:, :, np.newaxis], noise1[:, :, np.newaxis]),
                                       axis=2)).astype(np.uint8)
        noise2 = np.random.uniform(.1, .3, ((w - h) * w - (w - h) // 2 * w)).reshape([(w - h) - (w - h) // 2, w])
        noise2 = (255 * np.concatenate((noise2[:, :, np.newaxis], noise2[:, :, np.newaxis], noise2[:, :, np.newaxis]),
                                       axis=2)).astype(np.uint8)
        img_padded = np.vstack((noise1, img, noise2))
    else:
        img_padded = img

    img_resized = cv2.resize(img_padded, (size_H, size_W))

    return img_resized

def zero_fullbox_pad(img, size_H=256, size_W=256):
    '''
    pad zeros to make a square img for resize
    '''
    c, h, w = img.shape
    img_hwc = np.zeros((h,w,c))
    for i in range(3):
        img_hwc[:,:,i] = img[i,:,:]
    noise1 = np.zeros([h, int((size_W - w) / 2)])
    noise1 = (255 * np.concatenate((noise1[:, :, np.newaxis], noise1[:, :, np.newaxis], noise1[:, :, np.newaxis]),
                                   axis=2)).astype(np.uint8)
    noise11 = np.zeros([h, size_W -w-int((size_W - w) / 2)])
    noise11 = (255 * np.concatenate((noise11[:, :, np.newaxis], noise11[:, :, np.newaxis], noise11[:, :, np.newaxis]),
                                   axis=2)).astype(np.uint8)
    img_padded = np.hstack((noise1, img_hwc, noise11))
    noise2 = np.zeros([int((size_H-h)/2),size_W])
    noise2 = (255 * np.concatenate((noise2[:, :, np.newaxis], noise2[:, :, np.newaxis], noise2[:, :, np.newaxis]),
                                   axis=2)).astype(np.uint8)
    noise22 = np.zeros([size_H-h-int((size_H-h)/2),size_W])
    noise22 = (255 * np.concatenate((noise22[:, :, np.newaxis], noise22[:, :, np.newaxis], noise22[:, :, np.newaxis]),
                                   axis=2)).astype(np.uint8)
    img_padded = np.vstack((noise2, img_padded, noise22))

    img_resized = cv2.resize(img_padded, (128, 128))

    return img_resized

def read_avi(fname):
    cap = cv2.VideoCapture(fname)

    wid = int(cap.get(3))
    hei = int(cap.get(4))
    framerate = int(cap.get(5))
    framenum = int(cap.get(7))

    video = np.zeros((framenum, hei, wid, 3), dtype=np.uint8)
    c =0
    for i in range(framenum):
        try:
            a, b = cap.read()
            video[i] = b[..., ::-1]
            c+=1
        except:
            break
    video = video[:c,:,:,:]
    return video

def zero_pad_avi(vid, Sample_Bch):
    lenVid = len(vid)
    if lenVid>Sample_Bch:
        idx = np.linspace(0,lenVid-1,Sample_Bch).tolist()
        idx = [int(x) for x in idx]
        vid = vid[idx]
    elif lenVid < Sample_Bch:
        pad = np.zeros((Sample_Bch-lenVid,vid.shape[1],vid.shape[2],3), dtype=np.uint8)
        vid = np.concatenate([vid,pad],axis=0)
    return vid

def delay_pad_avi(vid, Sample_Bch):
    lenVid = len(vid)
    idx = np.linspace(0,lenVid-1,Sample_Bch).tolist()
    idx = [int(x) for x in idx]
    vid = vid[idx]

    return vid

def rgb2gray(rgb):
    x = np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    return np.rint(x)
HEIGHT =320
WIDTH = 320
def write_avi(video, vname, size = (WIDTH,HEIGHT)):
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter(vname,fourcc,30,size)
    for i in range(video.shape[0]):
        outVideo.write(video[i])
    outVideo.release()

def write_avi_g(video, vname, size = (WIDTH,HEIGHT)):
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
    outVideo = cv2.VideoWriter(vname,fourcc,30,size,isColor=False)
    for i in range(video.shape[0]):
        outVideo.write(video[i])
    outVideo.release()

if __name__ == '__main__':
    df1 = pd.read_excel(r'G:\甲状腺视频模型 数据及代码——伍凌鹄\patient_result.xlsx')
    df2 = pd.read_excel(r'G:\甲状腺视频模型 数据及代码——伍凌鹄\Thyroid_0714_filtered(1).xlsx')
    for i in range(len(df2)):
        if df2.loc[i, 'data_usage']=='test':
            if df2.loc[i, 'StudyID'] not in df1['StudyID']:
                df2.drop(index=i, inplace=True)
    df2.to_excel('G:\甲状腺视频模型 数据及代码——伍凌鹄\Thyroid_0714_filtered(2).xlsx')