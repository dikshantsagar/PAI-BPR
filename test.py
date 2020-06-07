from GPBPR2 import GPBPR
import torch

from torch.utils.data import TensorDataset,DataLoader
from torch import load, sigmoid, cat, rand, bmm, mean, matmul
from torch.nn.functional import logsigmoid
from torch.nn.init import uniform_

from torch.optim import Adam
from sys import argv
import pickle

## CHANGE
PATH = r"/content/drive/My Drive/GPBPR" ## parent directory of Group_6 folder

my_config = {
    "visual_features_dict": PATH + r"/Group_9/src/data/train/feat/visualfeatures",
    "textural_idx_dict": PATH + r"/Group_9/src/data/train/feat/textfeatures",
    "textural_embedding_matrix": PATH + r"/Group_9/src/data/train/feat/smallnwjc2vec",
    "train_data": PATH + r"/Group_9/src/data/train/data/train.csv",
    "test_data": PATH + r"/Group_9/src/data/train/data/test.csv",
    "model_file": PATH + r"/Group_9/src/data/train/model/final.model",
}

def load_csv_data(train_data_path):
    result = []
    with open(train_data_path,'r') as fp:
        for line in fp:
            t = line.strip().split(',')
            t = [int(i) for i in t]
            result.append(t)
    return result

def evaluating(model, mode, test_csv, visual_features, text_features):
    model.eval()
    testData = load_csv_data(test_csv)
    pos = 0
    batch_s = 100
    for i in range(0, len(testData), batch_s):
        data = testData[i:i+batch_s] if i+batch_s <=len(testData) else testData[i:]
        output = model.forward(data, visual_features,text_features)  
        #print(output.shape)
        pos += float(torch.sum(output.ge(0)))
    print( "Test Score: " , pos/len(testData))

def F(mode,  device):

    print('loading visual features')
    visual_features = torch.load(my_config['visual_features_dict'], map_location= lambda a,b:a.cpu())
    print('successful')
    text_features = torch.load(my_config['textural_idx_dict'], map_location= lambda a,b:a.cpu())

    print("loading model")
    gpbpr = load(my_config['model_file'], map_location=lambda x,y: x.cuda(device))
    print('successful') 
    
    evaluating(gpbpr,mode, my_config['test_data'], visual_features, text_features)


if __name__ == "__main__":

    # "cpu" or "cuda:x" x is GPU index like (0,1,2,3,)

    F(mode = 'final',  device = 0)
