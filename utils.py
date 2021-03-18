import sys
import os
import numpy
import cv2
import os
import json
import math
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import conf

def get_model(model_type, use_gpu):

    if model_type == 'vgg16':
        from models.vgg import vgg16
        model = vgg16()
    elif model_type == 'resnet50':
        from models.resnet import resnet50
        model = resnet50()
    elif model_type == 'resnet18':
        from models.resnet import resnet18
        model = resnet18()
    elif model_type == 'googlenet':
        from models.googlenet import googlenet
        model = googlenet()
    else:
        print('this model is not supported')
        sys.exit()
    
    if use_gpu:
        model = model.cuda()
    
    return model

def my_eval(model, data_path, use_gpu):
    model.eval()

    final_dict = {}
    json_dict = {}

    for filename in os.listdir(data_path):
        if filename.endswith('.mp4') or filename.endswith('.avi'):
            #continue
            video_path = data_path + '/' + filename
            #print(video_path)

            reader = cv2.VideoCapture(video_path)
            num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            ans = 0
            real_ans = 0
            fake_ans = 0
            all_poss = 0.0

            while reader.isOpened():
                _, image = reader.read()
                if image is None:
                    break
                
                ans += 1

                if ans % conf.FRAME_SAMPLE != 0:
                    continue

                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                image = image.resize((conf.IMAGE_SIZE, conf.IMAGE_SIZE),Image.ANTIALIAS) 
                transform = transforms.Compose([transforms.ToTensor()])
                img = transform(image)
                img = img.resize(1, 3, conf.IMAGE_SIZE, conf.IMAGE_SIZE)

                if use_gpu:
                    output = model(img.cuda())
                else:
                    output = model(img)
                all_poss += math.exp(output[0][0].cpu().detach().numpy()) / (math.exp(output[0][0].cpu().detach().numpy()) + math.exp(output[0][1].cpu().detach().numpy()))

                _, pred = output.topk(1, 1, True, True)

                if pred.cpu().numpy()[0][0] == 1:
                    real_ans += 1
                else:
                    fake_ans += 1
            
            if fake_ans + real_ans != 0:
                fake_poss = float(all_poss) / (fake_ans + real_ans)
            else:
                fake_poss = 0.5

            final_dict[filename] = fake_poss
            print(filename + ': ' + str(fake_poss))

        elif filename.endswith('.json'):
            print('found json')
            json_file = open(data_path + '/' + filename, encoding='utf-8')
            json_dict = json.load(json_file)
    
    # check
    correct_ans = 0
    score = 0
    for key in final_dict:
        if key in json_dict:
            if json_dict[key]['label'] == 'FAKE' if final_dict[key] > 0.5 else 'REAL':
                correct_ans += 1
            yi = 1 if json_dict[key]['label'] == 'FAKE' else 0
            if final_dict[key] >= 1:
                final_dict[key] = 0.999
            elif final_dict[key] <= 0:
                final_dict[key] = 0.001
            score += yi * math.log(final_dict[key]) + (1 - yi) * math.log(1 - final_dict[key])
        else:
            print('dict key unmatched')
        
    score = - score / len(final_dict)
    
    print('final accuracy:' + str(float(correct_ans) / len(final_dict)))
    print('final score: ' + str(score))

def resize_image(image, resize_height = 256, resize_width = 256):
    image_shape = numpy.shape(image)

    height = image_shape[0]
    width = image_shape[1]

    image = cv2.resize(image, dsize=(resize_width, resize_height))
    return image

def image_preprocess(filename, resize_height = 256, resize_width = 256, normalization=False):
    bgr_image = cv2.imread(filename)

    if bgr_image is None:
        print("Image does not exist: ", filename)
        return None
    if len(bgr_image.shape) == 2:
        print("Warning: gray image: ", filename)
        bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
 
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb_image = resize_image(rgb_image, resize_height, resize_width)
    rgb_image = numpy.asanyarray(rgb_image)
    if normalization:
        rgb_image = rgb_image / 255.0
    return rgb_image

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
