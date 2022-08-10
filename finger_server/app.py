
from codecs import latin_1_decode
from flask import Flask
from flask import request
from flask import render_template
import pathlib
import os
import model_fingerprint 
import EfficientNet_model
from tensorflow import keras
import resnet50_model1
import VGG16_model


app = Flask(__name__)

SRC_PATH =  pathlib.Path(__file__).parent.absolute()
UPLOAD_FOLDER = os.path.join(SRC_PATH,'static','uploads')


@app.route("/")
def page():
    return render_template("page1.html")

@app.route('/', methods=['POST'])
def upload_file():
    file = request.files['filename']         
    file.save(os.path.join(UPLOAD_FOLDER,"1.BMP"))
    return render_template('page1.html',A="1.BMP",B=file.filename)
                

@app.route("/page2",methods=["POST"])
def page2():
    sel = request.form["selname"]
    if sel =="1":  #T 符合對像 Q符合機率 T符合圖片名
        T1,Q1,P1=model_fingerprint.model1()
        Q1[0] = float(Q1[0])*100
        T5="Siamese networks"  
        model_fingerprint.img1(P1)     
        net=render_template("page2.html",T1=T1[0],T2=T1[1],T3=T1[2],T4=T1[3],Q1=str(Q1[0])+"%",T5=T5)
    if sel =="2":
        T2,Q2,P2=EfficientNet_model.model2()       
        for i in range(0,4):
            Q2[i] = round(float(Q2[i])*100,2)
        T5="EfficientNetB0"
        ID=str(Q2[0])+"%"
        gender=str(Q2[1])+"%"
        LRH=str(Q2[2])+"%"
        Finger=str(Q2[3])+"%"
        model_fingerprint.img1(P2)
        net=render_template("page3.html",T1=T2[0],T2=T2[1],T3=T2[2],T4=T2[3],
                            Q1=ID,Q2=gender,Q3=LRH,Q4=Finger,T5=T5)

    if sel =="4":
        T3,Q3,P3=resnet50_model1.model3()
        model_fingerprint.img1(P3) 
        for i in range(0,4):
            Q3[i] = round(float(Q3[i])*100,2)
        T5="ResNet50"
        ID=str(Q3[0])+"%"
        gender=str(Q3[1])+"%"
        LRH=str(Q3[2])+"%"
        Finger=str(Q3[3])+"%"
        model_fingerprint.img1(P3)
        net=render_template("page3.html",T1=T3[0],T2=T3[1],T3=T3[2],T4=T3[3],
                            Q1=ID,Q2=gender,Q3=LRH,Q4=Finger,T5=T5)
    if sel=="3":
        T4,Q4,P4=VGG16_model.model4()
        for i in range(0,4):
            Q4[i] = round(float(Q4[i])*100,2)
        T5="VGG 16"
        ID=str(Q4[0])+"%"
        gender=str(Q4[1])+"%"
        LRH=str(Q4[2])+"%"
        Finger=str(Q4[3])+"%"
        model_fingerprint.img1(P4)
        net=render_template("page3.html",T1=T4[0],T2=T4[1],T3=T4[2],T4=T4[3],
                            Q1=ID,Q2=gender,Q3=LRH,Q4=Finger,T5=T5)


    return net



if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)