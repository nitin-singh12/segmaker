import shutil
import os
import random
import pickle
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
# from backbone import Backbone
from model_irse import Backbone

from PIL import Image


class Knn_infer:
    def __init__(self, arcface_backbone,knn_model_path,labels_path):
        

        # KNN model
        KNN_MODEL_PATH = knn_model_path
        with open(KNN_MODEL_PATH, 'rb') as f:
            self.knn = pickle.load(f)
        
        LABEL_PATH = labels_path
        with open(LABEL_PATH) as json_file:
            dict1 = json.load(json_file)
            self.labels = dict1["labels"]

        # Embedding models
        self.EMBEDDING_SIZE = 512
        self.INPUT_SIZE =[224, 224]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.backbone  = arcface_backbone


  
    def get_embeddings(self,data_root):
        data_transform = transforms.Compose([transforms.Resize((self.INPUT_SIZE[0], self.INPUT_SIZE[1])), transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        image = Image.open(data_root)
        
        if image.mode == "L":
            image = image.convert("RGB")
        # r,g,b,a = image.split()
        # print(r,g,b,a)
        image = data_transform(image).unsqueeze(0).to(self.device)
        embeddings = F.normalize(self.backbone(image.to(self.device))).to(self.device)      
        return embeddings

    def infer_knn(self,input_dir):

        DATA_DIR = input_dir
        embeddings = self.get_embeddings(DATA_DIR)
        # with open("2020_embeddings/"+input_dir+".pickle", 'rb') as f:
        #     embeddings = pickle.load(f)  
        all_embeddings = np.array(embeddings.tolist())
        distances, indices = self.knn.kneighbors(all_embeddings,n_neighbors=5)
        y_pred = []
        class_list = []
        dist_list = []
        for i in range(0,len(indices)):
            temp = []   
            for j in range(0,len(indices[i])):
                temp.append(self.labels[indices[i][j]])
                class_list.append(self.labels[indices[i][j]])
                dist_list.append(distances[i][j])
            y_pred.append(max(temp,key=temp.count))
        list1 = y_pred[0].split("--")
        parallel = list1[0]
        series = list1[1]
        sets = list1[2]
        return y_pred[0]

    def inference(self,data_dir,outdir):
        classes = sorted(os.listdir(data_dir))
        gt_list = []
        pred_list = []
        d = []
        for i in range(0,len(classes)):
            images = os.listdir(data_dir+"/"+classes[i])
            count = 0
            total = len(images)
            for img in images:
                pred = self.infer_knn(data_dir+"/"+classes[i]+"/"+img)
                gt = classes[i]
                gt_list.append(gt)
                pred_list.append(pred)
                if gt==pred:
                    count = count+1
            val = count/total
            if val<0.5:
                d.append({"class_name":classes[i],"number":total-count})
        print("Accuracy : ",accuracy_score(gt_list,pred_list)*100)
        df = pd.DataFrame(d)
        if not os.path.exists(outdir+data_dir.split("/")[len(data_dir.split("/"))-2]):
            os.makedirs(outdir+data_dir.split("/")[len(data_dir.split("/"))-2])
        df.to_csv(outdir+data_dir.split("/")[len(data_dir.split("/"))-2]+"/"+data_dir.split("/")[-1]+".csv")    

