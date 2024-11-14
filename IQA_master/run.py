from __future__ import print_function, division
import os
import torch
from torch import nn
import pandas as pd
from skimage import transform
import numpy as np
#from torchsummary import summary
import torch.nn.functional as F

from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

import timm
from PIL import Image
import time
import math
import copy
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import one_kernel_p
import two_kernel_p
import three_kernel_p
import mergenet
import Resnet
import vit

import csv
import warnings
warnings.filterwarnings("ignore")
import random
from scipy.stats import spearmanr, pearsonr
use_gpu = True
Image.LOAD_TRUNCATED_IMAGES = True

ResultSave_path='record_fourkener_loss3_new_koniq.txt'

torch.backends.cudnn.benchmark = True

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
            #print(img_name)
            #print('ok')
            im = Image.open(img_name).convert('RGB')
            if im.mode == 'P':
                im = im.convert('RGB')
            image = np.asarray(im)
            #image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
            rating = self.images_frame.iloc[idx, 1]
            sample = {'image': image, 'rating': rating}

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
        image, rating = sample['image'], sample['rating']
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

        return {'image': image, 'rating': rating}


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
        image, rating = sample['image'], sample['rating']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'rating': rating}


class RandomHorizontalFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        if random.random() < self.p:
            image = np.flip(image, 1)
            # image = io.imread(img_name+'.jpg', mode='RGB').convert('RGB')
        return {'image': image, 'rating': rating}


class Normalize(object):
    def __init__(self):
        self.means = np.array([0.485, 0.456, 0.406])
        self.stds = np.array([0.229, 0.224, 0.225])

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']
        im = image /1.0#/ 255
        im[:, :, 0] = (image[:, :, 0] - 0.485) / 0.229
        im[:, :, 1] = (image[:, :, 1] - self.means[1]) / self.stds[1]
        im[:, :, 2] = (image[:, :, 2] - self.means[2]) / self.stds[2]
        image = im
        return {'image': image, 'rating': rating}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, rating = sample['image'], sample['rating']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'rating': torch.from_numpy(np.float64([rating])).double()}


class BaselineModel2(nn.Module):
    def __init__(self, num_classes, keep_probability, inputsize):

        super(BaselineModel2, self).__init__()
        self.fc1 = nn.Linear(inputsize, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.drop_prob = (1 - keep_probability)
        self.relu1 = nn.PReLU()
        self.drop1 = nn.Dropout(self.drop_prob)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.PReLU()
        self.drop2 = nn.Dropout(p=self.drop_prob)
        self.fc3 = nn.Linear(512, 25)
        self.bn3 = nn.BatchNorm1d(25)              #add norm
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
        self.fc3 = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)              #add norm
        self.sig = nn.Sigmoid()

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
        out = self.fc1(x)

        out = self.bn1(out)
        out = self.relu1(out)
        out = self.drop1(out)
        out = self.fc2(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.sig(out)
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

class Net(nn.Module):
    def __init__(self , net1, net2, net3, Vit, linear):
        super(Net, self).__init__()
        self.Net1 = net1
        self.Net2 = net2
        self.Net3 = net3
        self.vit = Vit
        self.Linear = linear
        #self.con = nn.Conv2d(2048, 768, kernel_size=1, stride=1, bias=False)
        
        self.c = nn.Conv2d(1280, 1024, kernel_size=1, stride=1, bias=False)
        
        #self.new = nn.Conv2d(1024, 768, kernel_size=(5,3), stride=(4,1), padding=1, bias=False)
        #self.new = nn.Conv2d(1024, 1024, kernel_size=(5,3), stride=(4,1), padding=1, bias=False)
        #self.new2 = nn.Conv2d(512, 768, kernel_size=1, stride=1, bias=False)
        self.threeDfusion = nn.Conv3d(16, 32, kernel_size=3, stride=(3, 1, 1), padding=(4, 1, 1), bias=False)
        
        self.new = nn.Conv2d(1024, 768, kernel_size=(5,3), stride=(3,1), padding=1, bias=False)
        
        self.atlinear1 = nn.Linear(1024, 768)
        #self.atlinear2 = nn.Linear(1024, 192)
        #self.atlinear3 = nn.Linear(1024, 192)
        #self.share_linear = nn.Linear(1024, 192)

        
    def feature_genetic1(self, x):
        cropped_tensors = []
        for i in range(1024):
            random_cropindex_h = np.random.randint(0, 7)
            random_cropindex_w = np.random.randint(0, 7)
            cropped_channel = x[:, i, random_cropindex_h : random_cropindex_h+7, random_cropindex_w : random_cropindex_w+7]
            cropped_tensors.append(cropped_channel)
        out = torch.stack(cropped_tensors, dim=1)
        return out
    
    '''
    def feature_genetic2(self, x):
        cropped_tensors = []
        for i in range(1024):
            random_cropindex_h = np.random.randint(0, 5)
            cropped_channel = x[:, i, random_cropindex_h : random_cropindex_h+9, :]
            cropped_tensors.append(cropped_channel)
        out = torch.stack(cropped_tensors, dim=1)
        return out
    '''
 
    def forward(self, x1, x2, x3, share_feature):
        x1, k1, k11 = self.Net1(x1)
        x2, k2 = self.Net2(x2)
        x3, k3 = self.Net3(x3)
        #print('x1', x1.shape)
        
        #x2 = F.avg_pool2d(x2, 5, stride=2, padding=2)
        x3 = F.avg_pool2d(x3, 5, stride=2, padding=2)
        x3 = self.c(x3)
        
        x1 = self.feature_genetic1(x1)
        x2 = self.feature_genetic1(x2)
        x3 = self.feature_genetic1(x3)
        share_feature = self.feature_genetic1(share_feature)
        
        #print(x1.shape)
        x_a = torch.cat((x1, x2), dim=2)
        x_b = torch.cat((x3, share_feature), dim=2)
        x = torch.cat((x_a, x_b), dim=3)
        
        bt_size = x.shape[0]
        x = x.reshape(bt_size, 16, 64, 14, 14)
        
        x = self.threeDfusion(x)
        x = x.reshape(bt_size, 768, 14, 14)
        #print(x.shape)
        '''
        x = rearrange(x, 'b c h w  -> b (h w) c ', h=14, w=14)
        #share_feature = rearrange(share_feature, 'b c h w  -> b (h w) c ', h=14, w=14)
        x = self.atlinear1(x)
        x = rearrange(x, 'b (h w) c  -> b c h w ', h=14, w=14)
        '''

        x = self.vit(x)
        x = self.Linear(x)
        return x, k1, k11, k2, k3



def computeSpearman(dataloader_valid1, dataloader_valid2, dataloader_valid3, model_help, model):
    ratings = []
    predictions = []
    with torch.no_grad():
        cum_loss = 0
        count = 0
        for data1, data2, data3 in zip(dataloader_valid1, dataloader_valid2, dataloader_valid3):
            inputs1 = data1['image']
            batch_size1 = inputs1.size()[0]
            labels1 = data1['rating'].view(batch_size1, -1)
            # labels = labels / 10.0
            inputs2 = data2['image']
            batch_size2 = inputs2.size()[0]
            labels2 = data2['rating'].view(batch_size2, -1)
            inputs3 = data3['image']
            batch_size3 = inputs3.size()[0]
            labels3 = data3['rating'].view(batch_size3, -1)

            if use_gpu:
                try:
                    inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                    inputs2 = Variable(inputs2.float().cuda())
                    inputs3 = Variable(inputs3.float().cuda())
                except:
                    print(inputs1, labels1, inputs2, labels2, inputs3, labels3)
            else:
                inputs1, labels1 = Variable(inputs1), Variable(labels1)
                inputs2 = Variable(inputs2)
                inputs3 = Variable(inputs3)
                
            x, share_feature = model_help(inputs2)

            #outputs_a, k1, k11, k2, k3 = model(inputs1, inputs1, inputs3, share_feature)
            outputs_a, k1, k11, k2, k3 = model(inputs1, inputs1, inputs3, share_feature)
            ratings.append(labels1.float())
            predictions.append(outputs_a.float())
            
            count += 1
            '''
            if count%10 == 0:
                print('k1 is:',k1, k11)
                print('k2 is:',k2)
                print('k3 is:',k3)
            '''

    ratings_i = np.vstack([r.detach().cpu().numpy() for r in ratings])
    predictions_i = np.vstack([p.detach().cpu().numpy() for p in predictions])
    a = ratings_i[:,0]
    b = predictions_i[:,0]
    sp = spearmanr(a, b)[0]
    pl = pearsonr(a,b)[0]
    return sp, pl

def finetune_model():
    epochs = 40
    srocc_l = []
    plcc_l = []
    epoch_record = []
    best_srocc = 0
    print('=============Saving Finetuned Prior Model===========')
    data_dir = os.path.join('/home/user/MetaIQA-master/LIVE_WILD/')
    images = pd.read_csv(os.path.join(data_dir, 'image_labeled_by_score.csv'), sep=',')
    images_fold = "/home/user/MetaIQA-master/LIVE_WILD/"
    if not os.path.exists(images_fold):
        os.makedirs(images_fold)
    for i in range(100):
        
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print(i,file=f1)

        
        print('\n')
        print('--------- The %2d rank trian-test (100epochs) ----------' % i )
        images_train, images_test = train_test_split(images, train_size = 0.4, test_size = 0.6)

        train_path = images_fold + "train_image" + ".csv"
        test_path = images_fold + "test_image" + ".csv"
        images_train.to_csv(train_path, sep=',', index=False)
        images_test.to_csv(test_path, sep=',', index=False)

        #model = torch.load('model_IQA/TID2013_IQA_Meta_resnet18-1.pt')
        net_1 = one_kernel_p.densenetnew(pretrained=False)
        net_2 = two_kernel_p.densenetnew(pretrained=False)
        net_3 = three_kernel_p.densenetnew(pretrained=False)
        #m_net = mergenet.merge_net(pretrained=False)
        l_net = BaselineModel1(1, 0.5, 1000)

        net_1_state_dict = net_1.state_dict()
        densenet_model = models.densenet121(pretrained = True)
        densenet_state_dict = densenet_model.state_dict()
        #state_dict = densenet_model.features.state_dict()
        common_state_dict = {k: v for k, v in densenet_state_dict.items() if k in net_1_state_dict}
        
        net_1_state_dict.update(common_state_dict)
        net_1.load_state_dict(net_1_state_dict)
    
        '''
        for name in list(state_dict.keys()):
            if name.startswith('denseblock4.'):
                del state_dict[name]
            if name.startswith('norm5.'):
                del state_dict[name]
            if name.startswith('transition3.'):
                del state_dict[name]
        #print(list(state_dict.keys()))
        net_1.features.load_state_dict(state_dict)
        #net_2.features.load_state_dict(state_dict)
        #net_3.features.load_state_dict(state_dict)
        '''
        
        pretrained_cfg_overlay = {'file': r"/home/user/use_trans/pytorch_model.bin"}
        vit_model = timm.create_model('vit_base_patch16_224', pretrained_cfg_overlay = pretrained_cfg_overlay ,pretrained=True)
        vmodel_state_dict = vit_model.state_dict()
        
        VIT = vit.VisionTransformer() 
        VIT_state_dict = VIT.state_dict()
        
        common_state_dict2 = {k: v for k, v in vmodel_state_dict.items() if k in VIT_state_dict}
        VIT_state_dict.update(common_state_dict2)
        '''
        state_dict_vit = VIT.state_dict()
        for name in list(state_dict_vit.keys()):
            print(name)
        '''
        VIT.load_state_dict(VIT_state_dict)

        model = Net(net1 = net_1, net2 = net_2, net3 = net_3, Vit = VIT, linear = l_net)
        
        resnet1 = models.resnet50(pretrained = True)
        net_re = Resnet.resnet50(pretrained=False)
        state_dict1 = resnet1.state_dict()

        net_re.load_state_dict(state_dict1)
        
        l_net2 = BaselineModel2(1, 0.5, 1000) 
        model_classify = Net1(net1 = net_re, linear = l_net2)
        model_classify.load_state_dict(torch.load('model_IQA/classify.pt'))
        
        
        '''
        input_size1 = (3, 224, 224)  # 输入1的尺寸
        input_size2 = (3, 224, 224)  # 输入2的尺寸
        input_size3 = (3, 224, 224)    # 输入3的尺寸

        summary(model, input_size=[input_size1, input_size2, input_size3])
        '''
        #model.load_state_dict(torch.load('model_IQA/TID2013_KADID10K_IQA_Meta_densenet_newload.pt'))

        for m in model.modules():
            if 'Conv' in str(type(m)):
                setattr(m, 'padding_mode', 'zeros')
        criterion = nn.MSELoss()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4,  weight_decay=0)
        model.cuda()
        model_classify.cuda()

        spearman = 0
        for epoch in range(epochs):
            optimizer = exp_lr_scheduler(optimizer, epoch)
            count = 0

            if epoch == 0:
                dataloader_valid1 = load_data('train1')
                dataloader_valid2 = load_data('train2')
                dataloader_valid3 = load_data('train1')
                model.eval()
                model_classify.eval()

                sp = computeSpearman(dataloader_valid1, dataloader_valid2, dataloader_valid3, model_classify, model)[0]
                if sp > spearman:
                    spearman = sp
                print('no train srocc {:4f}'.format(sp))

            # Iterate over data.
            #print('############# train phase epoch %2d ###############' % epoch)
            dataloader_train1 = load_data('train1')
            dataloader_train2 = load_data('train2')
            dataloader_train3 = load_data('train1')
            model.train()  # Set model to training mode
            model_classify.train()
            running_loss = 0.0
            total = 0
            for data1, data2, data3 in zip(dataloader_train1, dataloader_train2, dataloader_train3):
                inputs1 = data1['image']
                batch_size1 = inputs1.size()[0]
                labels1 = data1['rating'].view(batch_size1, -1)
                #print('input1', inputs1)
                # labels = labels / 10.0
                inputs2 = data2['image']
                batch_size2 = inputs2.size()[0]
                #labels2 = data2['rating'].view(batch_size2, -1)
                #print('input2', inputs2)
                inputs3 = data3['image']
                batch_size3 = inputs3.size()[0]
                #labels3 = data3['rating'].view(batch_size3, -1)
                #print('input3', inputs3)

                if use_gpu:
                    try:
                        inputs1, labels1 = Variable(inputs1.float().cuda()), Variable(labels1.float().cuda())
                        inputs2 = Variable(inputs2.float().cuda())
                        inputs3 = Variable(inputs3.float().cuda())
                    except:
                        print(inputs1, labels1, inputs2, labels2, inputs3, labels3)
                else:
                    inputs1, labels1 = Variable(inputs1), Variable(labels1)
                    inputs2 = Variable(inputs2)
                    inputs3 = Variable(inputs3)
                    
                x, share_feature = model_classify(inputs2)

                optimizer.zero_grad()
                outputs, k1, k11, k2, k3 = model(inputs1, inputs1, inputs3, share_feature)
                loss = criterion(outputs, labels1)
                loss.backward()
                optimizer.step()
                
                total += 1
                '''
                if total%5 == 0:
                    print('k1 is:',k1, k11)
                    print('k2 is:',k2)
                    print('k3 is:',k3)
                '''
                
                #print('t  e  s  t %.8f' %loss.item())
                try:
                    running_loss += loss.item()

                except:
                    print('unexpected error, could not calculate loss or do a sum.')

                count += 1

            epoch_loss = running_loss / count
            epoch_record.append(epoch_loss)
            print(' The %2d epoch : current loss = %.8f ' % (epoch,epoch_loss))

            #print('############# test phase epoch %2d ###############' % epoch)
            dataloader_valid1 = load_data('test1')
            dataloader_valid2 = load_data('test2')
            dataloader_valid3 = load_data('test1')
            model.eval()
            model_classify.eval()

            sp, pl = computeSpearman(dataloader_valid1, dataloader_valid2, dataloader_valid3, model_classify, model)
            if sp > spearman:
                spearman = sp
                plcc=pl
            if sp > best_srocc:
                best_srocc = sp
                print('=====Prior model saved===Srocc:%f========'%best_srocc)
                best_model = copy.deepcopy(model)
                torch.save(best_model.cuda(),'model_IQA/zhenghuo_vit.pt')

            print('Validation Results - Epoch: {:2d}, PLCC: {:4f}, SROCC: {:4f}, '
                  'best SROCC: {:4f}'.format(epoch, pl, sp, spearman))

        with open('livec_scores.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, plcc, spearman])

        srocc_l.append(spearman)
        plcc_l.append(pl)
        with open(ResultSave_path, 'a') as f1:  # 设置文件对象data.txt
            print('PLCC: {:4f}, SROCC: {:4f}'.format(plcc, spearman),file=f1)
    
    epoch_count = 0
    f = open('loss_record.txt','w')
    for line in epoch_record:
        epoch_record += 1
        f.write('epoch' + epoch_count + line + '\n')
        if epoch_record == 100:
            epoch_record = 0
    f.save()
    f.close()

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

    meta_num = 16
    data_dir = os.path.join('/home/user/MetaIQA-master/LIVE_WILD/')
    train_path = os.path.join(data_dir,  'train_image.csv')
    test_path = os.path.join(data_dir,  'test_image.csv')

    output_size = (224, 224)
    transformed_dataset_train1 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/LIVEwild/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(256, 256)),
                                                                                  RandomHorizontalFlip(0.5),
                                                                                  RandomCrop(
                                                                                      output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_train2 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/LIVEwild/salient_images/',
                                                    transform=transforms.Compose([RandomHorizontalFlip(0.5),
                                                                                  #RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_train3 = ImageRatingsDataset(csv_file=train_path,
                                                    root_dir='/home/user/data/LIVEwild/non_salient_images/',
                                                    transform=transforms.Compose([RandomHorizontalFlip(0.5),
                                                                                  #RandomCrop(output_size=output_size),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid1 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/LIVEwild/Images/',
                                                    transform=transforms.Compose([Rescale(output_size=(224, 224)),
                                                                                  Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid2 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/LIVEwild/salient_images/',
                                                    transform=transforms.Compose([Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    transformed_dataset_valid3 = ImageRatingsDataset(csv_file=test_path,
                                                    root_dir='/home/user/data/LIVEwild/non_salient_images/',
                                                    transform=transforms.Compose([Normalize(),
                                                                                  ToTensor(),
                                                                                  ]))
    bsize = meta_num

    if mod == 'train1':
        dataloader = DataLoader(transformed_dataset_train1, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)
    if mod == 'train2':
        dataloader = DataLoader(transformed_dataset_train2, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)
    if mod == 'train3':
        dataloader = DataLoader(transformed_dataset_train3, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)
    if mod == 'test1':
        dataloader = DataLoader(transformed_dataset_valid1, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)
    if mod == 'test2':
        dataloader = DataLoader(transformed_dataset_valid2, batch_size=bsize,
                                  shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)
    else:
        dataloader = DataLoader(transformed_dataset_valid3, batch_size=bsize,
                                    shuffle=False, num_workers=0, collate_fn=my_collate, drop_last=True)

    return dataloader

finetune_model()
