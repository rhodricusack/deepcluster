# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import math
import boto3
import botocore

import json

from util import AverageMeter, learning_rate_decay, load_model, Logger

import pandas as pd

import multiprocessing
multiprocessing.set_start_method('spawn', True)

parser = argparse.ArgumentParser(description="""Train linear classifier on top
                                 of frozen convolutional layers of an AlexNet.""")

parser.add_argument('--data', type=str, default='/imagenet',help='path to dataset')
# This stuff will be pulled from parameters pulled from SQS 
#parser.add_argument('--model', type=str, help='path to model')
#parser.add_argument('--conv', type=int, choices=[1, 2, 3, 4, 5],
#                    help='on top of which convolutional layer train logistic regression')
parser.add_argument('--checkpointbucket', type=str, default='neurana-imaging', help='bucket for checkpoint')
parser.add_argument('--checkpointpath', type=str, default='rhodricusack/deepcluster_analysis/checkpoints_2019-09-11/checkpoints', help='prefix on s3 for checkpoints')
parser.add_argument('--sqsurl', type=str, default='https://sqs.eu-west-1.amazonaws.com/807820536621/deepcluster-linearclass.fifo', help='SQS URL for task')

parser.add_argument('--linearclassbucket', type=str, default='neurana-imaging', help='bucket for linearclass')
parser.add_argument('--linearclasspath', type=str, default='rhodricusack/deepcluster_analysis/linearclass/', help='prefix on s3 for linearclass')

parser.add_argument('--tencrops', action='store_true',
                    help='validation accuracy averaged over 10 crops')
parser.add_argument('--aoaval', default=True, action='store_true',
                    help='age of acquisition style validation')
parser.add_argument('--exp', type=str, default='/home/ubuntu/linearclass_test', help='exp folder')
parser.add_argument('--workers', default=32, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', type=int, default=2, help='number of total epochs to run (default: 90)')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--weight_decay', '--wd', default=-4, type=float,
                    help='weight decay pow (default: -4)')
parser.add_argument('--seed', type=int, default=31, help='random seed')
parser.add_argument('--verbose', default=True, action='store_true', help='chatty')
parser.add_argument('--toplayer_epoch', type=int, default=None, help='top layer epoch to load up (default: None)')


def main():
    global args
    args = parser.parse_args()

    while True:
        #fix random seeds
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

        best_prec1 = 0

        # identify task
        client = boto3.client('sqs',region_name='eu-west-1')

        # Retry for a minute
        for retry in range(60):
            sqsreceive = client.receive_message(
                QueueUrl=args.sqsurl, MaxNumberOfMessages=1
            )
            if 'Messages' in sqsreceive.keys():
                break
            time.sleep(1.0)
            print('Retrying queue %s'%args.sqsurl)
        

        if not 'Messages' in sqsreceive.keys():
            print('No SQS found, bailing')
            return

        
        print('Received SQS:\n%s'%sqsreceive)


        # Parse message into model and conv
        msgbody=json.loads(sqsreceive['Messages'][0]['Body'])
        checkpointbasename='checkpoint_%d.pth.tar'%msgbody['epoch']
        checkpointfn=os.path.join(args.exp,checkpointbasename)
        conv=msgbody['conv']

        # Get rid of the message from the queue if we've got this far
        client.delete_message(ReceiptHandle=sqsreceive['Messages'][0]['ReceiptHandle'],QueueUrl=args.sqsurl)

        # Pull model from S3
        s3 = boto3.resource('s3')
        try:
            s3fn=os.path.join(args.checkpointpath, checkpointbasename)
            print("Attempting s3 download from %s"%s3fn )
            s3.Bucket(args.checkpointbucket).download_file(s3fn,checkpointfn)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                print("The object does not exist.")
            else:
                raise

        # Prepare place for output    
        linearclassfn=os.path.join(args.linearclasspath,"linearclass_time_%d_conv_%d"%(msgbody['epoch'],conv))
        print("Will write output to bucket %s, %s",args.linearclassbucket,linearclassfn)

        # load model
        model = load_model(checkpointfn)
        model.cuda()
        cudnn.benchmark = True

        # Recover disc
        os.remove(checkpointfn)

        # freeze the features layers
        for param in model.features.parameters():
            param.requires_grad = False

        # define loss function (criterion) and optimizer
        criterion = nn.CrossEntropyLoss().cuda()

        # data loading code
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val_in_folders')
        valdir_double = os.path.join(args.data,'val_in_double_folders')
        valdir_list=[]

        # Load in AoA table if needed
        if args.aoaval:
            aoalist=pd.read_csv('matchingAoA_ImageNet_excel.csv')
            for index,row in aoalist.iterrows():
                node=row['node']
                aoa=float(row['aoa'])
                if not math.isnan(aoa):
                    valdir_list.append({'node':node,'pth':os.path.join(valdir_double,node),'aoa':aoa})
                else:
                    print('Not found %s'% node)
                    
            #valdir_list=valdir_list[:5] trim for testing
            print('Using %d validation categories for aoa'%len(valdir_list))


        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        if args.tencrops:
            transformations_val = [
                transforms.Resize(256),
                transforms.TenCrop(224),
                transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])),
            ]
        else:
            transformations_val = [transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                normalize]

        transformations_train = [transforms.Resize(256),
                                transforms.CenterCrop(256),
                                transforms.RandomCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize]
        train_dataset = datasets.ImageFolder(
            traindir,
            transform=transforms.Compose(transformations_train)
        )

        val_dataset = datasets.ImageFolder(
            valdir,
            transform=transforms.Compose(transformations_val)
        )

        # Load up individual categories for AoA validation
        if args.aoaval:
            print("Loading individual categories for validation")
            val_list_dataset=[]
            val_list_loader=[]
            val_list_remap=[]
            for entry in valdir_list:
                val_list_dataset.append(datasets.ImageFolder(
                                        entry['pth'],
                                        transform=transforms.Compose(transformations_val)))

                val_list_loader.append(torch.utils.data.DataLoader(val_list_dataset[-1],
                                                batch_size=50,
                                                shuffle=False,
                                                num_workers=args.workers))
                val_list_remap.append(train_dataset.classes.index(entry['node']))

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.workers,
                                                pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=int(args.batch_size/2),
                                                shuffle=False,
                                                num_workers=args.workers)

        # logistic regression
        print("Setting up regression")

        reglog = RegLog(conv, len(train_dataset.classes)).cuda()
        optimizer = torch.optim.SGD(
            filter(lambda x: x.requires_grad, reglog.parameters()),
            args.lr,
            momentum=args.momentum,
            weight_decay=10**args.weight_decay
        )

        # create logs
        exp_log = os.path.join(args.exp, 'log')
        if not os.path.isdir(exp_log):
            os.makedirs(exp_log)

        loss_log = Logger(os.path.join(exp_log, 'loss_log'))
        prec1_log = Logger(os.path.join(exp_log, 'prec1'))
        prec5_log = Logger(os.path.join(exp_log, 'prec5'))



        # If savedmodel already exists, load this 
        print("Looking for saved decoder")
        if args.toplayer_epoch:
            filename="model_toplayer_epoch_%d.pth.tar"%args.toplayer_epoch
        else:
            filename='model_best.pth.tar'
        savedmodelpth=os.path.join(args.exp,filename)

        s3_client = boto3.client('s3')
        try:
            response = s3_client.download_file(args.linearclassbucket,os.path.join(linearclassfn,filename),savedmodelpth)
            print('Loading saved decoder %s'%savedmodelpth)
            model_with_decoder=torch.load(savedmodelpth)
            reglog.load_state_dict(model_with_decoder['reglog_state_dict'])
            lastepoch=model['epoch']+1
        except:
            lastepoch=0

        print("Will run from epoch %d to epoch %d"%(lastepoch,args.epochs-1))

        for epoch in range(lastepoch,args.epochs):
        # Top layer epochs
            end = time.time()
            # train for one epoch
            train(train_loader, model, reglog, criterion, optimizer, epoch)

            # evaluate on validation set
            prec1, prec5, loss = validate(val_loader, model, reglog, criterion,target_remap=range(1000))

            loss_log.log(loss)
            prec1_log.log(prec1)
            prec5_log.log(prec5)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            filename='model_toplayer_epoch_%d.pth.tar'%epoch
            
            modelfn=os.path.join(args.exp, filename)

            torch.save({
                'epoch': epoch + 1,
                'arch': 'alexnet',
                'state_dict': model.state_dict(),
                'reglog_state_dict': reglog.state_dict(),       # Also save decoding layers
                'prec5': prec5,
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, savedmodelpth)

            # Save output to check 
            s3_client = boto3.client('s3')
            response = s3_client.upload_file(savedmodelpth,args.linearclassbucket,os.path.join(linearclassfn,filename))
            for logfile in ['prec1','prec5','loss_log']:
                localfn=os.path.join(args.exp,'log',logfile)
                response = s3_client.upload_file(localfn,args.linearclassbucket,os.path.join(linearclassfn,'log',"%s_toplayer_epoch_%d"%(logfile,epoch)))

            if is_best:
                # Save output to check 
                s3_client = boto3.client('s3')
                response = s3_client.upload_file(savedmodelpth,args.linearclassbucket,os.path.join(linearclassfn,'model_best.pth.tar'))                
                for logfile in ['prec1','prec5','loss_log']:
                    localfn=os.path.join(args.exp,'log',logfile)
                    response = s3_client.upload_file(localfn,args.linearclassbucket,os.path.join(linearclassfn,'log',logfile))

            # Tidy up
            for logfile in ['prec1','prec5','loss_log']:
                localfn=os.path.join(args.exp,'log',logfile)
                os.remove(localfn)
            os.remove(savedmodelpth)

        if args.aoaval:
            # Validate individual categories, so loss can be compared to AoA

            # # To check weights loaded OK
            # # evaluate on validation set
            # prec1, prec5, loss = validate(val_loader, model, reglog, criterion)

            # loss_log.log(loss)
            # prec1_log.log(prec1)
            # prec5_log.log(prec5)

            aoares={}
            
            for idx,row in enumerate(zip(valdir_list,val_list_loader,val_list_remap)):
                # evaluate on validation set
                print("AOA validation %d/%d"%(idx,len(valdir_list)))
                prec1_tmp, prec5_tmp, loss_tmp = validate(row[1], model, reglog, criterion,target_remap=[row[2]])
                aoares[row[0]['node']]={'prec1':float(prec1_tmp),'prec5':float(prec5_tmp),'loss':float(loss_tmp),'aoa':row[0]['aoa']}
                

            # Save to JSON
            aoaresultsfn='aoaresults_toplayer_epoch_%d.json'%epoch
            aoapth=os.path.join(args.exp, aoaresultsfn)
            with open(aoapth,'w') as f:
                json.dump(aoares,f)
            response = s3_client.upload_file(aoapth,args.linearclassbucket,os.path.join(linearclassfn,aoaresultsfn))
            os.remove(aoapth)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False



class RegLog(nn.Module):
    """Creates logistic regression on top of frozen features"""
    def __init__(self, conv, num_labels):
        super(RegLog, self).__init__()
        self.conv = conv
        if conv==1:
            self.av_pool = nn.AvgPool2d(6, stride=6, padding=3)
            s = 9600
        elif conv==2:
            self.av_pool = nn.AvgPool2d(4, stride=4, padding=0)
            s = 9216
        elif conv==3:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==4:
            self.av_pool = nn.AvgPool2d(3, stride=3, padding=1)
            s = 9600
        elif conv==5:
            self.av_pool = nn.AvgPool2d(2, stride=2, padding=0)
            s = 9216
        self.linear = nn.Linear(s, num_labels)

    def forward(self, x):
        x = self.av_pool(x)
        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        return self.linear(x)


def forward(x, model, conv):
    if hasattr(model, 'sobel') and model.sobel is not None:
        x = model.sobel(x)
    count = 1
    for m in model.features.modules():
        if not isinstance(m, nn.Sequential):
            x = m(x)
        if isinstance(m, nn.ReLU):
            if count == conv:
                return x
            count = count + 1
    return x

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, reglog, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # freeze also batch norm layers
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #adjust learning rate
        learning_rate_decay(optimizer, len(train_loader) * epoch + i, args.lr)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target)
        # compute output

        output = forward(input_var, model, reglog.conv)
        output = reglog(output)
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
        
        
        

def validate(val_loader, model, reglog, criterion, target_remap=None):
    # Introduced target_remap, for use when the dataloader has not loaded all 1000 objects but a subset of them
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    softmax = nn.Softmax(dim=1).cuda()
    end = time.time()
    for i, (input_tensor, target) in enumerate(val_loader):
        if args.tencrops:
            bs, ncrops, c, h, w = input_tensor.size()
            input_tensor = input_tensor.view(-1, c, h, w)
        if target_remap:
            target=torch.tensor([target_remap[x] for x in target],dtype=torch.long)
        target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor.cuda())
            target_var = torch.autograd.Variable(target)

            output = reglog(forward(input_var, model, reglog.conv))

            if args.tencrops:
                output_central = output.view(bs, ncrops, -1)[: , ncrops / 2 - 1, :]
                output = softmax(output)
                output = torch.squeeze(output.view(bs, ncrops, -1).mean(1))
            else:
                output_central = output

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1[0], input_tensor.size(0))
            top5.update(prec5[0], input_tensor.size(0))
            loss = criterion(output_central, target_var)
            losses.update(loss.item(), input_tensor.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if args.verbose and i % 100 == 0:
                print('Validation: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                    .format(i, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))


    return top1.avg, top5.avg, losses.avg

if __name__ == '__main__':
    main()
