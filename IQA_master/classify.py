from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
import cv2

from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from torch.nn.parallel import DataParallel

torch.autograd.set_detect_anomaly(True)

from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import timm
import one_load
import mergenet
import Resnet_for_classif

import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

#torch.backends.cudnn.benchmark = True
ResultSave_path='record_freeze_vit.txt'

class ImageRatingsDataset(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating1 = self.images_frame.iloc[idx, 1]
            rating2 = self.images_frame.iloc[idx, 2]
            sample = {'image': image, 'rating1': rating1, 'rating2': rating2}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass


class ImageRatingsDataset2(Dataset):
    """Images dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.images_frame = pd.read_csv(csv_file, sep=',')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images_frame)

    def __getitem__(self, idx):
        # try:
            img_name = str(os.path.join(self.root_dir,str(self.images_frame.iloc[idx, 0])))
            im = Image.open(img_name).convert('L')
            if im.mode == 'P':
                im = im.convert('L')

            img_dct = cv2.dct(np.array(im, np.float32))  #get dct image

            image = np.asarray(img_dct)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating1': rating1, 'rating2': rating2}

            if self.transform:
                sample = self.transform(sample)
            return sample
        # except Exception as e:
        #     pass


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, rating1, rating2 = sample['image'], sample['rating1'], sample['rating2']
        #print(rating2)
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = transform.resize(image, (new_h, new_w))

        sample = {'image': image, 'rating1': rating1, 'rating2': rating2}
        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, rating1, rating2 = sample['image'], sample['rating1'], sample['rating2']
        #print(rating2)
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        sample = {'image': image, 'rating1': rating1, 'rating2': rating2}
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating1, rating2 = sample['image'], sample['rating1'], sample['rating2']
        #rating1 = sample['rating1']
        #print(rating1)
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        sample = {'image': image, 'rating1': rating1, 'rating2': rating2}
        return sample


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating1, rating2 = sample['image'], sample['rating1'], sample['rating2']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        sample = {'image': image, 'rating1': rating1, 'rating2': rating2}
        return sample



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating1, rating2 = sample['image'], sample['rating1'], sample['rating2']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating1': torch.from_numpy(np.float64([rating1])).double(),
                'rating2': torch.from_numpy(np.float64([rating2])).double()}

class BaselineModel1(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, 24)
        self.bn3 = nn.BatchNorm1d(24)              #add norm
        #self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        x = self.fc1(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        #print(out.shape)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        #out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out

class BaselineModel2(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel1, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, 1)
        self.bn3 = nn.BatchNorm1d(1)              #add norm
        #self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Weight initialization reference: https://arxiv.org/abs/1502.01852
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.02)
            #     m.bias.data.zero_()

    def forward(self, x):
        """
        Feed-forward pass.
        :param x: Input tensor
        : return: Output tensor
        """
        x = self.fc1(x)
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)
        #print(out.shape)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        #out = self.sig(out)
        # out_a = torch.cat((out_a, out_p), 1)

        # out_a = self.sig(out)
        return out
    
    

class Net1(nn.Module):
    def __init__(self, net1, linear):
        super(Net1, self).__init__()
        self.Net1 = net1
        self.Linear = linear

    def forward(self, x):
       
   
        x, share_feature = self.Net1(x)  #sa

        out = self.Linear(x)
        #print(out.shape)
        
        return out, share_feature
    
class Net2(nn.Module):
    def __init__(self, net1, net2, linear):
        super(Net2, self).__init__()
        self.Net1 = net1
        self.Net2 = net2
        self.Linear = linear

    def forward(self, x, x_share):
        #print(x1.shape)
        #print(x_share.shape)
   
        x = self.Net1(x)
        #print(x1.shape)
    
        #x1 = self.Net2(x1)  
        #print(x1.shape)
        
        features2 = torch.cat((x, x_share), dim=1)
        features2 = self.Net2(features2)
        out = self.Linear(features2)
        #print(out.shape)
        
        return out

'''
def computeSpearman_IQA(dataloader_valid1, model):
    ratings = []
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        cum_loss = 0
        for data1 in dataloader_valid1:
            inputs1 = data1['image']
            batch_size1 = inputs1.size()[0]
            labels1 = data1['rating1'].view(batch_size1, -1)
            labels2 = data1['rating2'].view(batch_size1, -1)

            if use_gpu:
                try:
                    inputs1, labels1, labels2 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda()), Variable(labels2.float().cuda())

                except:
                    print(inputs1, labels1, labels2)
            else:
                inputs1, labels1, labels2 = Variable(inputs1), Variable(labels1), Variable(labels2)

            outputs_a = model(inputs1)
            ratings.append(labels2.float())
            predictions.append(outputs_a.float())

    ratings_i = np.vstack([r.cpu().numpy() for r in ratings])
    predictions_i = np.vstack([p.cpu().numpy() for p in predictions])
    #ratings_i = np.vstack(ratings)
    #predictions_i = np.vstack(predictions)
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a,b)[0]
    return sp, pl    
'''
   
def computeSpearman_classify(dataloader_valid1, model1, model2):
    ratings = []
    predictions = []
    correct = 0
    total = 0
    with torch.no_grad():
        cum_loss = 0
        for data1 in dataloader_valid1:
            inputs1 = data1['image']
            batch_size1 = inputs1.size()[0]
            labels1 = data1['rating1'].view(batch_size1, -1)
            labels2 = data1['rating2'].view(batch_size1, -1)

            if use_gpu:
                try:
                    inputs1, labels1, labels2 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda()), Variable(labels2.float().cuda())

                except:
                    print(inputs1, labels1, labels2)
            else:
                inputs1, labels1, labels2 = Variable(inputs1), Variable(labels1), Variable(labels2)

            #####
            outputs_1, share_feature = model1(inputs1)
            _, predicted = torch.max(outputs_1, 1)
            #print('predicted:', predicted, predicted.shape)
            #print('labels1:', labels1, labels1.shape)
            predicted = predicted.squeeze()
            labels1 = labels1.squeeze()
            
            #print('predicted:', predicted, predicted.shape)
            #print('labels1:', labels1, labels1.shape)
            
            total += labels1.size(0)
            correct += (predicted == labels1).sum().item()
            #print(total)
            #print(correct)
            outputs_2 = model2(inputs1, share_feature)
            ratings.append(labels2.float())
            predictions.append(outputs_2.float())

        accuracy = 100 * correct / total
        
        ratings_i = np.vstack([r.cpu().numpy() for r in ratings])
        predictions_i = np.vstack([p.cpu().numpy() for p in predictions])
        #ratings_i = np.vstack(ratings)
        #predictions_i = np.vstack(predictions)
        a = ratings_i[:,0]
        b = predictions_i[:,0]
        sp = spearmanr(a, b)[0]
        pl = pearsonr(a,b)[0]
        #print('!!!', correct)
        #print(total)
    return accuracy, sp, pl

def finetune_model():
    epochs = 50
    srocc_l = []
    plcc_l = []
    epoch_record = []
    best_srocc = 0
    
    #torch.autograd.set_detect_anomaly(True)

    print('=============Saving Finetuned Prior Model===========')
    data_dir = os.path.join('/home/user/based_on_classification/KADID/')
    images = pd.read_csv(os.path.join(data_dir, 'classify_IQA.csv'), sep=',')
    images_fold = "/home/user/based_on_classification/KADID/"
    if not os.path.exists(images_fold):
        os.makedirs(images_fold)
    for i in range(10):
        best_predicted = 0
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print(i,file=f1)

        print('\n')
        print('--------- The %2d rank trian-test (24epochs) ----------' % i )
        images_train, images_test = train_test_split(images, train_size = 0.8)

        train_path = images_fold + "train_image" + ".csv"
        test_path = images_fold + "test_image" + ".csv"
        images_train.to_csv(train_path, sep=',', index=False)
        images_test.to_csv(test_path, sep=',', index=False)

        #resnet1 = models.resnet50(pretrained = True)
        
        net_N1 = one_load.densenetnew(pretrained=False)
        m_net = mergenet.merge_net(pretrained=False)
        densenet_model = models.densenet121(pretrained = True)
        state_dict = densenet_model.features.state_dict()

        for name in list(state_dict.keys()):
            #print(name)
            if name.startswith('denseblock4.'):
                del state_dict[name]
            if name.startswith('norm5.'):
                del state_dict[name]
        #print(list(state_dict.keys()))
        net_N1.features.load_state_dict(state_dict)
        
        resnet1 = models.resnet50(pretrained = True)
        net_1 = Resnet.resnet50(pretrained=False)
        state_dict1 = resnet1.state_dict()

        net_1.load_state_dict(state_dict1)
        
        
                        
        l_net = BaselineModel1(1, 0.5, 1000)  
        #net_1 = models.densenet121(pretrained = True)
        model_classify = Net1(net1 = net_1, linear = l_net)
        
        model_IQA = Net2(net1 = net_N1, net2 = m_net, linear = l_net)
        
        
        #model = torch.load('model_IQA/TID2013_IQA_Meta_resnet18-1.pt')
        '''
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}, Shape: {param.shape}")
        #model.load_state_dict(torch.load('model_IQA/TID2013_KADID10K_IQA_Meta_densenet_newload.pt'))
        #model.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
        '''

        for m in model_classify.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
             
        '''
        for param in model.Vit.parameters():
            param.requires_grad = False
        '''
        
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()

        optimizer1 = optim.Adam(model_classify.parameters(), lr=1e-4,  weight_decay=0)
        optimizer2 = optim.Adam(model_IQA.parameters(), lr=1e-4,  weight_decay=0)
        model_classify.cuda()  
        model_IQA.cuda()
        #model = DataParallel(model)
        
        spearman = 0
        pre_ = 0
        for epoch in range(epochs):
            
            optimizer2 = exp_lr_scheduler(optimizer2, epoch)
            count = 0

            if epoch == 0:
                dataloader_valid1 = load_data('train1')

                model_classify.eval()

                #pre1 = computeSpearman_classify(dataloader_valid1, model_classify)
                #pre2 = computeSpearman_IQA(dataloader_valid1, model_IQA)[0]
                pre1, pre2, pre3 = computeSpearman_classify(dataloader_valid1, model_classify, model_IQA)
                if pre1 > best_predicted:
                    best_predicted1 = pre1
                    best_predicted2 = pre2
                print('no train curracy {:4f}'.format(pre1))
                print('no train  srocc {:4f}'.format(pre2))

            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train1 = load_data('train1')

            model_classify.train()
            model_IQA.train()# Set model to training mode
            running_loss = 0.0
            for data1 in dataloader_train1:
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating1'].view(batch_size1, -1)
                labels2 = data1['rating2'].view(batch_size1, -1)

                if use_gpu:
                    try:
                        inputs1, labels1, labels2 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda()), Variable(labels2.float().cuda())
 
                    except:
                        print(inputs1, labels1)
                else:
                    inputs1, labels1, labels2 = Variable(inputs1), Variable(labels1), Variable(labels2)

                #print(labels1.shape)
                inputs2 = inputs1
                
                optimizer1.zero_grad()
                outputs1, share_feature1 = model_classify(inputs1)
                labels1 = labels1.squeeze()
                loss1 = criterion1(outputs1, labels1.long())
                loss1.backward()
                optimizer1.step()
                #print('1')
                
                optimizer2.zero_grad()
                share_f = share_feature1.detach().clone()
                outputs2 = model_IQA(inputs2, share_f)
                loss2 = criterion2(outputs2, labels2)
                loss2.backward()
                optimizer2.step()
                #print('2')
                
                '''
                #optimizer2.zero_grad()
                #outputs2 = model_IQA(inputs1, share_f)
                #labels2 = labels2.squeeze()
                loss2 = criterion2(outputs2, labels2)
                loss2.backward()
                optimizer2.step()
                '''
                
                #print('t  e  s  t %.8f' %loss.item())
                try:
                    running_loss += loss1.item()

                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                count += 1

            epoch_loss = running_loss / count
            epoch_record.append(epoch_loss)
            print(' The %2d epoch : current loss = %.8f ' % (epoch,epoch_loss))

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid1 = load_data('test1')

            model_classify.eval()
            model_IQA.eval()
            
            predicted, sp, pl = computeSpearman_classify(dataloader_valid1, model_classify, model_IQA)
            
            if predicted > best_predicted:
                best_predicted = predicted
                print('=====Prior model saved===predicted:%f========'%best_predicted)
                best_model = copy.deepcopy(model_classify)
                torch.save(best_model.cuda(),'model_IQA/classify.pt')
            
            print('Validation Results - Epoch: {:2d}, predicted: {:4f}, best_predicted: {:4f}, '
                  .format(epoch, predicted, best_predicted))
            
            if sp > spearman:
                spearman = sp
                plcc=pl
            if sp > best_srocc:
                best_srocc = sp
                print('=====Prior model saved===Srocc:%f========'%best_srocc)
                best_model = copy.deepcopy(model_IQA)
                #if best_srocc > 0.94:
                torch.save(best_model.cuda(),'model_IQA/IQA_clasiify.pt')

            print('Validation Results - Epoch: {:2d}, PLCC: {:4f}, SROCC: {:4f}, '
                  'best SROCC: {:4f}'.format(epoch, pl, sp, spearman))


    '''
    epoch_count = 0
    f = open('loss_record.txt','w')
    for line in epoch_record:
        epoch_record += 1
        f.write('epoch' + epoch_count + line + '\n')
        if epoch_record == 100:
            epoch_record = 0
    f.save()
    f.close()
    '''
    # ind = 'Results/LIVEWILD'
    # file = pd.DataFrame(columns=[ind], data=srocc_l)
    # file.to_csv(ind+'.csv')
    # print('average srocc {:4f}'.format(np.mean(srocc_l)))

def exp_lr_scheduler(optimizer, epoch, lr_decay_epoch=10):
    """Decay learning rate by a factor of DECAY_WEIGHT every lr_decay_epoch epochs."""

    decay_rate =  0.8**(epoch // lr_decay_epoch)
    if epoch % lr_decay_epoch == 0:
        print('decay_rate is set to {}'.format(decay_rate))

    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate

    return optimizer

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


def load_data(mod = 'train1'):

    meta_num = 24
    data_dir = os.path.join('/home/user/based_on_classification/KADID/')
    train_path = os.path.join(data_dir,  'train_image.csv')
    test_path = os.path.join(data_dir,  'test_image.csv')

    output_size = (224, 224)
                                               
    transformed_dataset_train = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/KADID/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    
    transformed_dataset_valid = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/KADID/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    
    bsize = meta_num

    if mod == 'train1':
        dataloader = DataLoader(transformed_dataset_train, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    
    if mod == 'test1':
        dataloader = DataLoader(transformed_dataset_valid, batch_size=bsize,
                                  shuffle=False, num_workers=4, collate_fn=my_collate)
    

    return dataloader

finetune_model()
