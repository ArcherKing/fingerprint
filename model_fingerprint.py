import numpy as np
from tensorflow import keras
import cv2

def restore_label(label):
    finger_list = ['thumb', 'index', 'middle', 'ring', 'little']
    label_list = [i for i in label]
    label_list[1] = 'F' if label_list[1] else 'M'
    label_list[2] = 'Right' if label_list[2] else 'Left'
    label_list[3] = finger_list[label_list[3]]
    return label_list

# Load Dataset
x_real = np.load('dataset/x_real.npz')['data']
y_real = np.load('dataset/y_real.npy')
x_easy = np.load('dataset/x_easy.npz')['data']
y_easy = np.load('dataset/y_easy.npy')
x_medium = np.load('dataset/x_medium.npz')['data']
y_medium = np.load('dataset/y_medium.npy')
x_hard = np.load('dataset/x_hard.npz')['data']
y_hard = np.load('dataset/y_hard.npy')

# Make Label Dictionary Lookup Table
label_real_dict = {}

for i, y in enumerate(y_real):
    key = y.astype(str)
    key = ''.join(key).zfill(6)

    label_real_dict[key] = i
len(label_real_dict)

# load model
dir(keras.models)
model =  keras.models.load_model('./data/fingerprint_220727.h5')

# 需鑑識指紋處理
img_path = "./uploads/89__M_Right_middle_finger_CR.BMP"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (96, 96))
np_img = np.array(img).reshape((1, 96, 96, 1))
np_img = np.array(list(np_img)*len(x_real))
np_img = np_img.astype(np.float32) / 255.

# 指紋比對
pred = model.predict([np_img, x_real.astype(np.float32) / 255.])
probability_list = np.argsort(pred, axis=0)

print('輸入指紋:', img_path)
print('符合對象:', restore_label(y_real[probability_list[-1]][0]))
print('符合機率:', pred[probability_list[-1]][0])

