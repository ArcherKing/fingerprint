import cv2
import matplotlib.pyplot as plt
import numpy as np

import glob, os

def extract_label(img_path):
    filename, _ = os.path.splitext(os.path.basename(img_path))
    
    subject_id, etc = filename.split('__')
    gender, lr, finger, _ = etc.split('_')
    
    gender = 0 if gender == 'M' else 1
    lr = 0 if lr =='Left' else 1
    
    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
        
    return np.array([subject_id, gender, lr, finger], dtype=np.uint16)

def extract_label2(img_path):

    filename, _ = os.path.splitext(os.path.basename(img_path))
    
    subject_id, etc = filename.split('__')
    gender, lr, finger, _, _ = etc.split('_')
    
    gender = 0 if gender == 'M' else 1
    lr = 0 if lr =='Left' else 1
    
    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
        
    return np.array([subject_id, gender, lr, finger], dtype=np.uint16)

datapaths = {'real':'SOCOFing/Real/*.BMP',
            'easy':'SOCOFing/Altered/Altered-Easy/*.BMP',
            'medium':'SOCOFing/Altered/Altered-Medium/*.BMP',
            'hard':'SOCOFing/Altered/Altered-Hard/*.BMP'}

for key, path in datapaths.items():
    img_list = sorted(glob.glob(path))
    # print(len(img_list))

    imgs = np.empty((len(img_list), 96, 96, 3), dtype=np.uint8)
    labels = np.empty((len(img_list), 4), dtype=np.uint16)

    for i, img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (96, 96))
        imgs[i] = img
        
        if key in 'real':
            # subject_id, gender, lr, finger
            labels[i] = extract_label(img_path)
        elif key in 'easy, medium, hard':
            labels[i] = extract_label2(img_path)
        
    np.savez('dataset_c/x_{}.npz'.format(key), data=imgs)
    np.save('dataset_c/y_{}.npy'.format(key), labels)

    plt.figure(figsize=(1, 1))
    plt.title(labels[-1])
    plt.imshow(imgs[-1])