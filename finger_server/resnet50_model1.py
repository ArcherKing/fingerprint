
import tensorflow as tf

import keras

import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input




x_real = np.load('dataset_c/x_real.npz')['data']
y_real = np.load('dataset_c/y_real.npy')
x_easy = np.load('dataset_c/x_easy.npz')['data']
y_easy = np.load('dataset_c/y_easy.npy')
x_medium = np.load('dataset_c/x_medium.npz')['data']
y_medium = np.load('dataset_c/y_medium.npy')
x_hard = np.load('dataset_c/x_hard.npz')['data']
y_hard = np.load('dataset_c/y_hard.npy')




print(x_real.shape,y_real.shape)
print(x_easy.shape,y_easy.shape)
print(x_medium.shape,y_medium.shape)
print(x_hard.shape,y_hard.shape)




def restore_label(label):
    finger_list = ['thumb', 'index', 'middle', 'ring', 'little']
    label_list = [i for i in label]
    label_list[1] = '女性' if label_list[1] else '男性'
    label_list[2] = '右手' if label_list[2] else '左手'
    label_list[3] = finger_list[label_list[3]]
    return label_list




def model3():
    # one-hot
    id_label = to_categorical(y_real[:,0]-1)
    gender_label = to_categorical(y_real[:,1])
    LRhand_label = to_categorical(y_real[:,2])
    finger_label = to_categorical(y_real[:,3])




# print(type(id_label))
# print(id_label.shape)
# print(gender_label.shape)
# print(LRhand_label.shape)
# print(finger_label.shape)




    # load model
    rs_model = ResNet50(include_top=False, weights="imagenet",input_shape=(120,120,3))


    id_model =  tf.keras.models.load_model('./data/resnet50_fpAll_id.h5')
    gender_model =  tf.keras.models.load_model('./data/resnet50_fpAll_gneder.h5')
    LRhand_model =  tf.keras.models.load_model('./data/resnet50_fpAll_LR.h5')
    finger_model =  tf.keras.models.load_model('./data/resnet50_fpAll_finger.h5')

    img_path = "./static/uploads/1.BMP"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (120, 120))
    np_img = np.array(img).reshape((1, 120, 120, 3))
    np_img = np_img.astype(np.float32) / 255.

    input = preprocess_input(np_img)
    features = rs_model.predict(input, verbose=0)

    # 指紋比對
    id_pred = id_model.predict(features)
    id_prob_list = np.argsort(id_pred[0], axis=0)

    gender_pred = gender_model.predict(features)
    gender_prob_list = np.argsort(gender_pred[0], axis=0)

    LRhand_pred = LRhand_model.predict(features)
    LR_prob_list = np.argsort(LRhand_pred[0], axis=0)

    finger_pred = finger_model.predict(features)
    finger_prob_list = np.argsort(finger_pred[0], axis=0)

    T1= restore_label([id_prob_list[-1]+1, gender_prob_list[-1], LR_prob_list[-1], finger_prob_list[-1]]) #符合對像
    Q1=[id_pred[0][id_prob_list[-1]], gender_pred[0][gender_prob_list[-1]], LRhand_pred[0][LR_prob_list[-1]], finger_pred[0][finger_prob_list[-1]]] #符合機率
    P1="" #符合圖片名
        
    if T1[0]!="":
        P1=P1+str(T1[0])
    if T1[1]=="男性":
        P1=P1+"__M"
    else:
        P1=P1+"__F"
    if T1[2]=="右手":
        P1=P1+"_Right"
    else:
        P1=P1+"_Left"
    P1=P1+"_"+T1[3]+"_finger.BMP"
    return T1,Q1,P1



#print('輸入指紋:', img_path)
#print('符合對象:', restore_label([id_prob_list[-1]+1, gender_prob_list[-1], LR_prob_list[-1], finger_prob_list[-1]]))
#print('符合機率:', [id_pred[0][id_prob_list[-1]], gender_pred[0][gender_prob_list[-1]], LRhand_pred[0][LR_prob_list[-1]], finger_pred[0][finger_prob_list[-1]]])

