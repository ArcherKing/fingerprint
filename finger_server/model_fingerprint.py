import numpy as np
from tensorflow import keras
import cv2
import keras
from keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

def restore_label(label):
    finger_list = ['thumb', 'index', 'middle', 'ring', 'little']
    label_list = [i for i in label]
    label_list[1] = '女性' if label_list[1] else '男性'
    label_list[2] = '右手' if label_list[2] else '左手'
    label_list[3] = finger_list[label_list[3]]
    return label_list

def model1():
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
    
    # load model
    model =  keras.models.load_model('./data/fingerprint_220727.h5')

    # 需鑑識指紋處理
    img_path = "./static/uploads/1.BMP"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (96, 96))
    np_img = np.array(img).reshape((1, 96, 96, 1))
    np_img = np.array(list(np_img)*len(x_real))
    np_img = np_img.astype(np.float32) / 255.

    # 指紋比對
    pred = model.predict([np_img, x_real.astype(np.float32) / 255.])
    probability_list = np.argsort(pred, axis=0)

    T1= restore_label(y_real[probability_list[-1][0]]) #符合對像
    Q1=pred[probability_list[-1]][0] #符合機率
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


def img1(PX):
    P2="../finger_server/SOCOFing/Real/"+str(PX)
    img_path = "./static/uploads/2.BMP"
    try:
        img = cv2.imread(P2)
        cv2.imwrite(img_path,img)
    except:
        img = cv2.imread("../finger_server/SOCOFing/Real/error.png")
        cv2.imwrite(img_path,img)

    

#print('輸入指紋:', img_path)
#print('符合對象:', Q1[0])
#print('符合機率:', pred[probability_list[-1]][0])

# Load Dataset
