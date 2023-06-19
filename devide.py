import cv2
import os
import numpy as np
import random
path = "D:/work/PatternRecognition/dataset/"
face_detect = cv2.CascadeClassifier('C:/Users/Wen HL/AppData/Local/Programs\
/Python/Python311/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
trainpath=path+"train/"
testpath=path+"test/"
for i in range(22,30):
    if not os.path.exists(trainpath+str(i)):
        os.mkdir(trainpath+str(i))
    if not os.path.exists(testpath+str(i)):
        os.mkdir(testpath+str(i))
    folder=path+str(i)+"/"
    file_names = os.listdir(folder)
    for file in file_names:
        imgpath = folder+file
        img=cv2.imread(imgpath)
        img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        img=np.array(img,"uint8")
        faces=face_detect.detectMultiScale(img,1.2,6)
        file=file.split('.')[0]
        for x,y,w,h in faces:
            temprand=random.random()
            if temprand<0.75:
                cv2.imwrite(trainpath+str(i)+"/"+file+".jpg",cv2.resize(img[y:y+h,x:x+w],dsize=(300,400)))
            else:
                cv2.imwrite(testpath+str(i)+"/"+file+".jpg",cv2.resize(img[y:y+h,x:x+w],dsize=(300,400)))
            temprand=random.random()
            if temprand<0.75:
                cv2.imwrite(trainpath+str(i)+"/"+file+"_flip.jpg",cv2.resize(cv2.flip(img[y:y+h,x:x+w],1),dsize=(300,400)))
            else :
                cv2.imwrite(testpath+str(i)+"/"+file+"_flip.jpg",cv2.resize(cv2.flip(img[y:y+h,x:x+w],1),dsize=(300,400)))
            mean = 0
            sigma = 25
            gauss = np.random.normal(mean,sigma,(400,300))
            noisy_img = cv2.resize(cv2.flip(img[y:y+h,x:x+w],1),dsize=(300,400)) + gauss
            noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
            temprand=random.random()
            if temprand<0.75:
                cv2.imwrite(trainpath+str(i)+"/"+file+"_noise.jpg",noisy_img)
            else :
                cv2.imwrite(testpath+str(i)+"/"+file+"_noise.jpg",noisy_img)