import shutil
import os
import random
import pickle
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
# from backbone import Backbone
# from model_irse import Backbone

from PIL import Image

class KNN:
    def __init__(self,arcface_backbone,out_path = "/home/ec2-user/SageMaker/all_knn/"):
        
        self.EMBEDDING_SIZE = 512
        self.INPUT_SIZE =[224, 224]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        
        self.knn_save_path = out_path
        self.backbone  = arcface_backbone

    def get_embeddings(self,data_root):
        data_transform = transforms.Compose([transforms.Resize((self.INPUT_SIZE[0], self.INPUT_SIZE[1])), transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        image = Image.open(data_root)
        image = data_transform(image).unsqueeze(0).to(self.device)
        embeddings = F.normalize(self.backbone(image.to(self.device))).to(self.device)      
        return embeddings

    def find_knn(self,all_embeddings,all_labels):
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(all_embeddings, all_labels)
        return classifier


    def generate_knn(self,input_dir):
        category  = sorted(os.listdir(input_dir))
        all_embeddings = []
        all_labels = []
        for i in range(len(category)):
            images = os.listdir(input_dir+"/"+category[i])
            for img in images:
                embeddings = self.get_embeddings(input_dir+"/"+category[i]+"/"+img)
                all_embeddings.append(embeddings.tolist())
                all_labels.append(category[i])
        print("all embeddings ",len(all_embeddings))
        print("all labels ",len(all_labels))

        all_embeddings = np.vstack(all_embeddings)
        all_labels = np.array(all_labels)

        out = self.find_knn(all_embeddings,all_labels)
        if not os.path.exists(self.knn_save_path+str(input_dir.split("/")[len(input_dir.split("/"))-2])+"/"+str(input_dir.split("/")[len(input_dir.split("/"))-1])):
            os.makedirs(self.knn_save_path+str(input_dir.split("/")[len(input_dir.split("/"))-2])+"/"+str(input_dir.split("/")[len(input_dir.split("/"))-1]))
        with open(self.knn_save_path+str(input_dir.split("/")[len(input_dir.split("/"))-2])+"/"+str(input_dir.split("/")[len(input_dir.split("/"))-1])+"/"+"KNN_"+str(input_dir.split("/")[-1])+".pickle", 'wb') as f:
            pickle.dump(out, f)
    
        final_dict = {"labels":all_labels.tolist()}
        with open(self.knn_save_path+str(input_dir.split("/")[len(input_dir.split("/"))-2])+"/"+str(input_dir.split("/")[len(input_dir.split("/"))-1])+"/"+"all_labels_"+str(input_dir.split("/")[-1])+".json", "w") as outfile:
            json.dump(final_dict, outfile)




# if __name__ == "__main__":
    
    

