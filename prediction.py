import argparse
import cv2
import json
import numpy as np
import os
import pandas as pd
import scipy.misc
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torchvision
import torchvision.models as models
import utils
import requests

from PIL import Image
from averagemeter import *
from models import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.autograd import Variable
from torch.utils.data import sampler
from torchvision import datasets
from torchvision import transforms
from skimage.transform import resize
from skimage.util import crop 
    
# GLOBAL CONSTANTS
INPUT_SIZE = 224
NUM_CLASSES = 185
USE_CUDA = torch.cuda.is_available()
best_prec1 = 0
saved_model = "leafnet/saved_models/model_best_augment.pth.tar"
testdir = 'C:/Users/Niranjan khedkar/OneDrive/Desktop/leafnet with model/leafnet/dataset_all/leafsnap/dataset/testit'

def download_image(pic_url, image_loc = 'C:/Users/Niranjan khedkar/OneDrive/Desktop/leafnet with model/leafnet/dataset_all/leafsnap/dataset/testit/unknown/example.jpg'):

    with open(image_loc, 'wb') as handle:
        response = requests.get(pic_url, stream=True)

        if not response.ok:
            print(response)

        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)
    return image_loc

def resize_image(image_loc, res = 224):
    size =res,res
    
    

    im = Image.open(image_loc)
    width, height = im.size   # Get dimensions
    dim = min(width, height)
    new_width = dim; new_height = dim
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    im = im.crop((left, top, right, bottom))
    im.thumbnail(size)
    im.save(image_loc)
    
    
def test(image_loc,test_loader, model, criterion, classes):
    # switch to evaluate mode
    model.eval()
    transform = transforms.Compose([
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    img = Image.open(image_loc)
    img_tensor = transform(img).unsqueeze(0)
    # for i, (input, target) in enumerate(test_loader):
    with torch.no_grad(): 
        # input_var = torch.autograd.Variable(input)
        output = model(img_tensor)

    #     confidences, predicted = torch.max(output.cpu().data, 1)
    # labels = classes[predicted.item()]
    # confidences = int(confidences.item())
    
        yhat = np.array(torch.nn.Softmax(dim=1)(output.cpu().data))[0]
        print(yhat)
        # yhat[yhat<0.3] = 0.
        counts = len(yhat[yhat>0])
        print(counts)
        # if counts == 0:
        #     labels = []; confidences = []
        # else:
        labels = yhat.argsort()[-counts:][::-1]
        labels = [classes[labels[0]]]
        confidences = 100*np.sort(yhat)[-counts:][::-1]
        confidences = confidences.astype(int)
            
    return labels, confidences

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 185)
    
    criterion = nn.CrossEntropyLoss()
    if USE_CUDA:
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
    
    if os.path.isfile(saved_model):
        checkpoint = torch.load(saved_model, map_location = 'cpu')
        best_prec1 = checkpoint['best_prec1']
        
        state_dict = checkpoint['state_dict']
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict,strict=False)        
    else:
        print("=> no checkpoint found at '{}'".format(saved_model))
    
    #print('\n[INFO] Reading Training and Testing Dataset')
    traindir = "C:/Users/Niranjan khedkar/OneDrive/Desktop/leafnet with model/leafnet/dataset_all/leafsnap/dataset/train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_train = datasets.ImageFolder(traindir)
    data_test = datasets.ImageFolder(testdir, transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ]))
    classes = data_train.classes
    
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, num_workers=0)
    
    return test_loader, model, criterion, classes
    
    
def predict_leaf(pic_url):
    image_loc = download_image(pic_url)
    resize_image(image_loc)
    test_loader, model, criterion, classes = setup_model()
    
    labels, confidences = test(image_loc,test_loader, model, criterion, classes)
    return labels, confidences
def main():
    labels, confidences = predict_leaf('http://www.tree-guide.com/images/styles/600x450-copy_/public/yellow-buckeye-leaf-underside.jpg?itok=UMkI9com')
    print(labels, confidences)
if __name__ == '__main__':
    main()