#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
from termios import VMIN
import time
from xml.dom import minicompat
import numpy as np
import yaml
import pickle
from collections import OrderedDict
import csv
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import shutil
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import random
import inspect
import torchmetrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandbFunctions as wandbF
import wandb
import time
from data_gen.getConnectingPoint import *

wandbFlag = True

# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self):
#         super(LabelSmoothingCrossEntropy, self).__init__()
#     def forward(self, x, target, smoothing=0.1):
#         confidence = 1. - smoothing
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = confidence * nll_loss + smoothing * smooth_loss
#         return loss.mean()


model_name = ''
def create_one_folder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def create_folder(directory):
    path = directory.split('/')
    total_path =''
    for i in path:
        total_path = os.path.join(total_path,i)
        #print(i, '  create : ',total_path)
        create_one_folder(total_path)
    
    #print('directory : ',directory)
    create_one_folder(directory)
    create_one_folder(directory+'/')
    #print('created paths')

def init_seed(value_seed):
    torch.cuda.manual_seed_all(value_seed)
    torch.manual_seed(value_seed)
    np.random.seed(value_seed)
    random.seed(value_seed)
    #torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Decoupling Graph Convolution Network with DropGraph Module')
    parser.add_argument('--work-dir',default='./work_dir/temp',help='the work folder for storing results')

    parser.add_argument('-model_saved_directory', default='')
    parser.add_argument('-experiment_name', default='')
    parser.add_argument('--config',default='config/sign/train/train_joint.yaml',help='path to the configuration file')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score',type=str2bool,default=False,help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval',type=int,default=100,help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval',type=int,default=2,help='the interval for storing models (#iteration)')
    parser.add_argument('--eval-interval',type=int,default=5,help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log',type=str2bool,default=True,help='print logging or not')
    parser.add_argument('--show-topk',type=int,default=[1, 5],nargs='+',help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker',type=int,default=32,help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args',default=dict(),help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args',default=dict(),help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args',type=dict,default=dict(),help='the arguments of model')
    parser.add_argument('--weights',default=None,help='the weights for network initialization')
    parser.add_argument('--ignore-weights',type=str,default=[],nargs='+',help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base_lr', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--num_epoch',type=int,default=500,help='stop training in which epoch')
    
    parser.add_argument('--step',type=int,default=[20, 40, 60],nargs='+',help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--device',type=int,default=0,nargs='+',help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=32, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch',type=int,default=0,help='start training from which epoch')
    parser.add_argument('--weight-decay',type=float,default=0.0001,help='weight decay for optimizer')
    parser.add_argument('--keep_rate',type=float,default=0.9,help='keep probability for drop')
    parser.add_argument('--groups',type=int,default=8,help='decouple groups')
    parser.add_argument('--only_train_part', default=True)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    
    # Data
    
    parser.add_argument("--experiment_name", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--training_set_path", type=str, default="", help="Path to the training dataset CSV file")
    parser.add_argument("--keypoints_model", type=str, default="openpose", help="Path to the training dataset CSV file")
    parser.add_argument("--keypoints_number", type=int, default=29, help="Path to the training dataset CSV file")
    parser.add_argument("--testing_set_path", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--num_class", type=int, default=0, help="Path to the testing dataset CSV file")
    parser.add_argument("--database", type=str, default="", help="Path to the testing dataset CSV file")
    parser.add_argument("--mode_train", type=str, default="train", help="Path to the testing dataset CSV file")

    return parser


class Processor():
    """ 
        Processor for Skeleton-based Action Recgnition
    """

    def __init__(self, arg):


        self.arg = arg
        self.save_arg()
        self.connectingPoints(arg)


        if arg.phase == 'train':
            pass
            '''
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_directory):
                    print('log_dir: ', arg.model_saved_directory, 'already exist')
                    answer = 'y'#input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_directory)
                        print('Dir removed: ', arg.model_saved_directory)
                        #input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_directory)
            '''

        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_tmp_acc = 0

        self.maxTestAcc = 0
        self.relative_maxtop5 = 0


    def connectingPoints(self,arg):
        print('Creating points .. ')

        folderName= '1' # just used to create folder "1" in data/sign/1/
        out_folder='data/sign/'
        out_path = os.path.join(out_folder, folderName)

        kp_model = arg.kp_model# 'wholepose' # openpose wholepose mediapipe
        dataset = arg.experiment_name# "PUCP" # WLASL PUCP_PSL_DGI156 AEC
        numPoints = arg.keypoints_number # number of points used, need to be: 29 or 71
        data_path_train = arg.training_set_path #f'../../../../joe/ConnectingPoints/split/WLASL--wholepose-Train.hdf5'
        data_path_test = arg.testing_set_path#f'../../../../joe/ConnectingPoints/split/WLASL--wholepose-Val.hdf5'
            

        model_key_getter = {'mediapipe': get_mp_keys,
                            'openpose': get_op_keys,
                            'wholepose': get_wp_keys}

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        print('kp_model',kp_model)
        print('\n',kp_model, dataset,'\n')
        print(out_path,'->', 'train')
        gendata(data_path_train, out_path, model_key_getter[kp_model], part='train', config=numPoints)
        print(out_path,'->', 'val')
        gendata(data_path_test, out_path, model_key_getter[kp_model], part='val', config=numPoints)
        print('Creating points completed!!! ')


    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        ln = Feeder(**self.arg.test_feeder_args)
        self.meaning = ln.meaning
        #print(ln.meaning)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device =  self.arg.device[0] if type(
             self.arg.device) is list else self.arg.device
        self.output_device =  output_device
        
        print('^'*20)
        print('self.arg.model',self.arg.model)
        print('model_args',self.arg.model_args)

        Model = import_class(self.arg.model)


        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
        if wandbFlag:
            wandbF.watch(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)

        path_model_init =  os.path.join(arg.model_saved_directory,arg.kp_model + '-' + arg.database +'-'+str(arg.keypoints_number)+ "-"+str(arg.seed)+"-init.pt")

        self.print_log('%'*20)
        self.print_log('path_model_init :')
        self.print_log(path_model_init)
        torch.save(self.model.state_dict(), path_model_init)
        self.print_log('%'*20)

        # self.loss = LabelSmoothingCrossEntropy().cuda(output_device)


        #self.slrt_model_wp.load_state_dict(self.slrt_model_op.state_dict())
        
        if self.arg.weights:
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)  
            else:
                weights = torch.load(self.arg.weights)  
                self.print_log("weights readed!")

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            for w in self.arg.ignore_weights:
                if weights.pop(w, None) is not None:
                    self.print_log('Sucessfully Remove Weights: {}.'.format(w))
                else:
                    self.print_log('Can Not Remove Weights: {}.'.format(w))

            try:
                self.print_log("load state dict weights")
                self.model.load_state_dict(weights)
                self.print_log("load state dict weights completed!")

            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):

        if self.arg.optimizer == 'SGD':

            params_dict = dict(self.model.named_parameters())
            params = []

            for key, value in params_dict.items():
                decay_mult = 0.0 if 'bias' in key else 1.0

                lr_mult = 1.0
                weight_decay = 1e-4

                params += [{'params': value, 'lr': self.arg.base_lr, 'lr_mult': lr_mult,
                            'decay_mult': decay_mult, 'weight_decay': weight_decay}]
            if wandbFlag:
                wandb.config = {
                    "learning_rate": self.arg.base_lr,
                    "epochs": self.arg.num_epoch,
                    "batch_size": self.arg.batch_size,
                    "weight_decay":self.arg.weight_decay,
                    "num_class":self.arg.model_args["num_class"],
                    "momentum":0.9
                }
            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                nesterov=self.arg.nesterov)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)

            if wandbFlag:
                wandb.config = {
                    "learning_rate": self.arg.base_lr,
                    "epochs": self.arg.num_epoch,
                    "batch_size": self.arg.batch_size,
                    "weight_decay":self.arg.weight_decay,
                    "num_class":self.arg.model_args["num_class"]
                }
        else:
            raise ValueError()

        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1,
                                              patience=10, verbose=True,
                                              threshold=1e-4, threshold_mode='rel',
                                              cooldown=0)


    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
            os.makedirs(self.arg.work_dir + '/eval_results')
        os.makedirs(self.arg.work_dir + '/eval_results/'+ model_name, exist_ok = True)

        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)


    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()


    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)


    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)


    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time


    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train_zero(self, epoch, save_model=False): 
        self.model.train(False)
        loader = self.data_loader['train']
        loss_value = []
        predict_arr = []
        proba_arr = []
        target_arr = []

        self.record_time()

        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        meaning = list(self.meaning.values())
        
        for batch_idx, (data, label, index, name) in enumerate(process):

            self.global_step += 1

            label_tmp = label.cpu().numpy()
            # get data
            data = Variable(data.float().cuda(self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            if epoch < 100:
                keep_prob = -(1 - self.arg.keep_rate) / 100 * epoch + 1.0
            else:
                keep_prob = self.arg.keep_rate

            output = self.model(data, keep_prob)

            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0

            #print('output',output)
            #print('label',label)
            loss = self.loss(output, label)
            #print('loss',loss)
            #for r,s in zip(name,label_tmp):
            #    meaning[s]= '_'.join(r.split('_')[:-1])

            loss_value.append(loss.data.cpu().numpy())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)

            predict_arr.append(predict_label.cpu().numpy())
            target_arr.append(label.data.cpu().numpy())
            proba_arr.append(output.data.cpu().numpy())

            acc = torch.mean((predict_label == label.data).float())


            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                        batch_idx, len(loader), loss.data, self.lr))
            timer['statistics'] += self.split_time()
        
        predict_arr = np.concatenate(predict_arr)
        target_arr = np.concatenate(target_arr)
        proba_arr = np.concatenate(proba_arr)
        accuracy  = torch.mean((predict_label == label.data).float())
        if accuracy >= self.best_tmp_acc:
            self.best_tmp_acc = accuracy

        if epoch+1 == arg.num_epoch:
            if wandbFlag:
                wandb.log({"TRAIN_conf_mat" : wandb.plot.confusion_matrix(
                            #probs=score,
                            #y_true=list(label.values()),
                            #preds=list(predict_label.values()),
                            y_true=list(target_arr),
                            preds=list(predict_arr),
                            class_names=meaning,
                            title="TRAIN_conf_mat")})

        if wandbFlag:
            mean_loss = np.mean(loss_value)
            if mean_loss>10:
                mean_loss = 10
            wandbF.wandbTrainLog(mean_loss, accuracy)
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
            

    def train(self, epoch, save_model=False): 
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        loss_value = []
        predict_arr = []
        proba_arr = []
        target_arr = []

        self.record_time()

        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)
        if epoch >= self.arg.only_train_epoch:
            print('only train part, require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = True
                    print(key + '-require grad')
        else:
            print('only train part, do not require grad')
            for key, value in self.model.named_parameters():
                if 'DecoupleA' in key:
                    value.requires_grad = False
                    print(key + '-not require grad')

        meaning = list(self.meaning.values())
        
        for batch_idx, (data, label, index, name) in enumerate(process):

            self.global_step += 1

            label_tmp = label.cpu().numpy()
            # get data
            data = Variable(data.float().cuda(
                self.output_device), requires_grad=False)
            label = Variable(label.long().cuda(
                self.output_device), requires_grad=False)
            timer['dataloader'] += self.split_time()

            # forward
            if epoch < 100:
                keep_prob = -(1 - self.arg.keep_rate) / 100 * epoch + 1.0
            else:
                keep_prob = self.arg.keep_rate

            output = self.model(data, keep_prob)

            if isinstance(output, tuple):
                output, l1 = output
                l1 = l1.mean()
            else:
                l1 = 0
            loss = self.loss(output, label) + l1
            
            #for r,s in zip(name,label_tmp):
            #    meaning[s]= '_'.join(r.split('_')[:-1])

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.cpu().numpy())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)

            predict_arr.append(predict_label.cpu().numpy())
            target_arr.append(label.data.cpu().numpy())
            proba_arr.append(output.data.cpu().numpy())

            acc = torch.mean((predict_label == label.data).float())

            self.lr = self.optimizer.param_groups[0]['lr']

            if self.global_step % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{:.6f}'.format(
                        batch_idx, len(loader), loss.data, self.lr))
            timer['statistics'] += self.split_time()
        
        predict_arr = np.concatenate(predict_arr)
        target_arr = np.concatenate(target_arr)
        proba_arr = np.concatenate(proba_arr)
        accuracy  = torch.mean((predict_label == label.data).float())
        if accuracy >= self.best_tmp_acc:
            self.best_tmp_acc = accuracy

        if epoch+1 == arg.num_epoch:
            if wandbFlag:
                wandb.log({"TRAIN_conf_mat" : wandb.plot.confusion_matrix(
                            #probs=score,
                            #y_true=list(label.values()),
                            #preds=list(predict_label.values()),
                            y_true=list(target_arr),
                            preds=list(predict_arr),
                            class_names=meaning,
                            title="TRAIN_conf_mat")})

        if wandbFlag:
            mean_loss = np.mean(loss_value)
            if mean_loss>10:
                mean_loss = 10

            wandbF.wandbTrainLog(mean_loss, accuracy)
        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
            

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None, isTest=False):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        #if isTest:
        submission = dict()
        trueLabels = dict()
        
        meaning = list(self.meaning.values())
        self.model.eval()
        with torch.no_grad():
            self.print_log('Eval epoch: {}'.format(epoch + 1))
            for ln in loader_name:

                loss_value = []
                score_frag = []
                right_num_total = 0
                total_num = 0
                loss_total = 0
                step = 0
                process = tqdm(self.data_loader[ln])

                for batch_idx, (data, label, index, name) in enumerate(process):
                    label_tmp = label
                    data = Variable(
                        data.float().cuda(self.output_device),
                        requires_grad=False)
                    label = Variable(
                        label.long().cuda(self.output_device),
                        requires_grad=False)

                    with torch.no_grad():
                        output = self.model(data)

                    if isinstance(output, tuple):
                        output, l1 = output
                        l1 = l1.mean()
                    else:
                        l1 = 0

                    #print('val output',output)
                    #print('val label',label)
                    loss = self.loss(output, label)
                    #print('val loss',loss)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.cpu().numpy())

                    _, predict_label = torch.max(output.data, 1) 

                    #if isTest:
                    for j in range(output.size(0)):
                        submission[name[j]] = predict_label[j].item()
                        trueLabels[name[j]] = label_tmp[j].item()

                    step += 1

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' +
                                        str(x) + ',' + str(true[i]) + '\n')
                score = np.concatenate(score_frag)

                if 'UCLA' in arg.experiment_name:
                    self.data_loader[ln].dataset.sample_name = np.arange(
                        len(score))

                accuracy = self.data_loader[ln].dataset.top_k(score, 1)
                top5 = self.data_loader[ln].dataset.top_k(score, 5)
                
                if accuracy > self.best_acc:
                    self.best_acc = accuracy

                    score_dict = dict(
                        zip(self.data_loader[ln].dataset.sample_name, score))

                    conf_mat = torchmetrics.ConfusionMatrix(num_classes=self.arg.model_args["num_class"])
                    '''
                    print('self.arg.model_args["num_class"]',self.arg.model_args["num_class"])
                    
                    print('list(submission.values())',list(submission.values()))
                    print('set(list(submission.values()))',set(list(submission.values())))
                    print('len(set(list(submission.values())))',len(set(list(submission.values()))))

                    print('list(trueLabels.values())',list(trueLabels.values()))
                    print('set(list(trueLabels.values()))',set(list(trueLabels.values())))
                    print('len(set(list(trueLabels.values())))',len(set(list(trueLabels.values()))))
                    '''
                    confusion_matrix = conf_mat(torch.tensor(list(submission.values())).cpu(), torch.tensor(list(trueLabels.values())).cpu())
                    confusion_matrix = confusion_matrix.detach().cpu().numpy()
                    
                    plt.figure(figsize = (10,7))
        
                    group_counts  = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
                    '''
                    print('confusion_matrix')
                    print(confusion_matrix)
                    print('len confusion_matrix')

                    print(len(confusion_matrix))
                    for line in confusion_matrix:
                        print('line',line)
                        print(len(line))
                    '''
                    confusion_matrix = np.asarray([line/(np.sum(line)+0.0001) for line in confusion_matrix])
                    confusion_matrix = np.nan_to_num(confusion_matrix)

                    df_cm = pd.DataFrame(confusion_matrix * 100, index = meaning, columns=meaning)
                    #size_arr = df_cm.sum(axis = 1)
                    #maxi = max(size_arr)

                    group_percentages = ["{0:.1%}".format(value) for value in confusion_matrix.flatten()]
                    
                    annot = ["{1}".format(v2, v1) for v1, v2 in zip(group_counts, group_percentages)]
                    annot = np.asarray(annot).reshape(self.arg.model_args["num_class"], self.arg.model_args["num_class"])
                    fig_ = sns.heatmap(df_cm, vmax=100, vmin=0, annot=annot, annot_kws={"size": 5}, cbar_kws={'format': '%.0f%%', 'ticks':[0, 25, 50, 75, 100]},fmt='', cmap='Blues').get_figure()
                    plt.ylabel('True label')
                    plt.xlabel('Predicted label' )
                    
                    plt.close(fig_)

                    if wandbFlag:
                        wandb.log({"Confusion matrix": wandb.Image(fig_, caption="VAL_conf_mat")})

                    
                    print('*'*20)
                    print('*'*20)
                    print('*'*20)

                    print('./work_dir/' + arg.experiment_name + '/eval_results/'+ model_name+ '/best_acc' + '.pkl')


                    with open('./work_dir/' + arg.experiment_name + '/eval_results/'+ model_name+ '/best_acc' + '.pkl'.format(
                            epoch, accuracy), 'wb') as f:
                        pickle.dump(score_dict, f)

                    # Save the model
                    state_dict = self.model.state_dict()
                    weights = OrderedDict([[k.split('module.')[-1],
                                        v.cpu()] for k, v in state_dict.items()])

                    print('*'*20)
                    print('*'*20)
                    print('*'*20)
                    print(self.arg.model_saved_directory)
                    print(self.arg.model_saved_directory + '-' + arg.kp_model + '-' + arg.database + "-Lr" + str(arg.base_lr) + "-NClasses" + str(arg.model_args["num_class"]) + '-' + str(accuracy) + '.pt')
                    torch.save(weights, self.arg.model_saved_directory + '-' + arg.kp_model + '-' + arg.database + "-Lr" + str(arg.base_lr) + "-NClasses" + str(arg.model_args["num_class"]) + '-' + str(accuracy) + '.pt')

                
                if epoch + 1 == arg.num_epoch:

                    if wandbFlag:
                        try:
                            wandb.log({"roc" : wandb.plot.roc_curve( list(trueLabels.values()), score, \
                                    labels=meaning, classes_to_plot=None)})
                    
                            wandb.log({"pr" : wandb.plot.pr_curve(list(trueLabels.values()), score,
                                    labels=meaning, classes_to_plot=None)})
                        except:
                            pass
                    #wandb.log({"val_sklearn_conf_mat": wandb.sklearn.plot_confusion_matrix(, 
                    #        , meaning_3)})
                    '''
                    wandb.log({"VAL_conf_mat" : wandb.plot.confusion_matrix(
                        #probs=score,
                        y_true=list(trueLabels.values()),
                        preds=list(submission.values()),
                        class_names=meaning_3,
                        title="VAL_conf_mat")})
                    '''

                print('Eval Accuracy: ', accuracy,
                    ' model: ', self.arg.model_saved_directory)
                if wandbFlag:
                    mean_loss = np.mean(loss_value)
                    if mean_loss>10:
                        mean_loss = 10

                    self.maxTestAcc = max(accuracy,self.maxTestAcc)
                    
                    if self.maxTestAcc == accuracy:

                        self.relative_maxtop5 = top5

                    wandbF.wandbValLog(mean_loss, accuracy, top5,self.maxTestAcc,self.relative_maxtop5)

                score_dict = dict(
                    zip(self.data_loader[ln].dataset.sample_name, score))
                self.print_log('\tMean {} loss of {} batches: {}.'.format(
                    ln, len(self.data_loader[ln]), np.mean(loss_value)))
                for k in self.arg.show_topk:
                    self.print_log('\tTop{}: {:.2f}%'.format(
                        k, 100 * self.data_loader[ln].dataset.top_k(score, k)))
                '''
                with open('./work_dir/' + arg.experiment_name + '/eval_results/epoch_' + str(epoch) + '_' + str(accuracy) + '.pkl'.format(
                        epoch, accuracy), 'wb') as f:
                    pickle.dump(score_dict, f)
                '''

        
        predLabels = []
        groundLabels = []
        print("END")
        if isTest:
            #print(submission)
            #print(trueLabels)
            totalRows = 0
            with open("submission.csv", 'w') as of:
                writer = csv.writer(of)
                accum = 0
                for trueName, truePred in trueLabels.items():

                    sample = trueName
                    #print(f'Predicting {sample}', end=' ')
                    #print(f'as {submission[sample]} - pred {submission[sample]} and real {row[1]}')
                    match=0
                    predLabels.append(submission[sample])
                    groundLabels.append(int(truePred))
                    if int(truePred) == int(submission[sample]):
                        match=1
                        accum+=1
                    totalRows+=1
                    
                    # identifying subject
                    with open("pucpSubject.csv") as subjectFile:
                        readerSubject = csv.reader(subjectFile)
                        idx = int(sample.split('_')[-1])
                        subjectName = 'NA'
                        for name, idxStart, idxEnd in readerSubject:
                            if (int(idxStart) <= idx) and (idx<= int(idxEnd)):
                                subjectName = name
                                break
                    writer.writerow([sample, submission[sample], str(truePred), str(match), subjectName])

        return np.mean(loss_value)


    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * \
                len(self.data_loader['train']) / self.arg.batch_size

            self.model.train(False)
            self.train_zero(0, save_model=False)
            val_loss = self.eval(0,save_score=self.arg.save_score,loader_name=['test'])
            self.model.train(True)

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                save_model = ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)

                self.train(epoch, save_model=save_model)

                val_loss = self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])

                # self.lr_scheduler.step(val_loss)

            print('best accuracy: ', self.best_acc,
                  ' model_name: ', self.arg.model_saved_directory)

        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_directory + '_wrong.txt'
                rf = self.arg.model_saved_directory + '_right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=self.arg.start_epoch, save_score=self.arg.save_score,
                      loader_name=['test'], wrong_file=wf, result_file=rf, isTest=True)
            self.print_log('Done.\n')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    

    parser = get_parser()
    arg = parser.parse_args()
    print('seed :',arg.seed)
    init_seed(arg.seed)

    for id_iteration in range(1):

        # load arg form config file



        print('arg.config',arg.config)
        if arg.config is not None:
            with open(arg.config, 'r') as f:
                #default_arg = yaml.load(f)
                default_arg = yaml.safe_load(f)
                print('default_arg',default_arg)
            key = vars(arg).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('WRONG ARG: {}'.format(k))
                    assert (k in key)
            parser.set_defaults(**default_arg)
            
        # load arg form config file
        arg = parser.parse_args()

        arg.training_set_path = '../../../dataset_original/'+arg.database+'--'+arg.keypoints_model+'-Train.hdf5'
        arg.testing_set_path  = '../../../dataset_original/'+arg.database+'--'+arg.keypoints_model+'-Val.hdf5'

        if arg.database == 'AEC':
            arg.num_class  = 28 

        if arg.database == 'WLASL':

            arg.num_class  = 86 
                
        if arg.database == 'PUCP':
            arg.num_class  = 29 
            arg.training_set_path = '../../../dataset_original/PUCP_PSL_DGI156--'+arg.keypoints_model+'-Train.hdf5'
            arg.testing_set_path  = '../../../dataset_original/PUCP_PSL_DGI156--'+arg.keypoints_model+'-Val.hdf5'

        arg.model_args['num_class'] =arg.num_class
        arg.model_args['num_point'] =arg.keypoints_number

        arg.model_args['graph_args']['num_node'] =arg.keypoints_number

        #num_class: 28 # AEC=28, PUCP=36 , WLASL=101
        #num_point: 29 # 29 or 71

        # arg.training_set_path
        # arg.keypoints_model
        # arg.keypoints_number
        # arg.testing_set_path
        # arg.experiment_name
        # arg.base_lr
        # arg.num_epoch

        
        config = {
                #
                "num-epoch": arg.num_epoch,
                "weight-decay": arg.weight_decay,
                "batch-size":arg.batch_size,
                "base-lr":  arg.base_lr,
                "kp-model": arg.keypoints_model,
                "num_points": arg.keypoints_number,
                "database": arg.database,
                "mode_train":arg.mode_train,
                "seed":arg.seed,
                "id_iteration":id_iteration,
        }

        if wandbFlag:
            wandb.init(project="sign_language_project", 
                    entity="ml_projects",
                    config=config)

            config = wandb.config
        print('+'*10)
        print('config :',config)
        print('+'*10)
        arg.base_lr = config["base-lr"]
        arg.batch_size = config["batch-size"]
        arg.weight_decay = config["weight-decay"]
        arg.num_epoch = config["num-epoch"]
        arg.kp_model = config["kp-model"]
        arg.database = config["database"]

        arg.model_saved_directory = "save_models/"+arg.experiment_name+"/"
        arg.work_dir              = "work_dir/"+arg.experiment_name+"/"

        print('*'*20)
        print('*'*20)

        print('model_saved_directory',arg.model_saved_directory)
        print('work_dir',arg.work_dir)


        create_folder(arg.model_saved_directory)
        create_folder(arg.work_dir)
        create_folder('./work_dir/' + arg.experiment_name + '/eval_results/'+ model_name+ '/')

        # {arg.model_saved_directory}-{arg.kp_model}-{arg.database}-Lr{str(arg.base_lr)}-NClasses{str(arg.num_class)}-{str(config['num_points'])}
        #os.makedirs(arg.file_name,exist_ok=True)

        runAndModelName =  arg.kp_model + '-' + arg.database +'-'+str(arg.keypoints_number)+ "-Lr" + str(arg.base_lr)+ "-NClas" + str(arg.num_class) + "-Batch" + str(arg.batch_size)+"-Seed"+str(arg.seed)+"-id"+str(id_iteration)

        model_name = runAndModelName
        print('model_name : ',model_name)
        if wandbFlag:
            wandb.run.name = runAndModelName
            wandb.run.save()



        print("*"*30)
        print("*"*30)
        print(arg)
        print("*"*30)
        print("*"*30)
        print(arg.train_feeder_args)
        print('train_feeder_args',arg.train_feeder_args)
        processor = Processor(arg)
        processor.start()
        if wandbFlag:
            wandb.finish()
            print("wandb finish")
