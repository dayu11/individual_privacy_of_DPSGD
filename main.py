import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv
import random
import time
import numpy as np

from models import resnet20
from utils import process_grad_batch, get_grad_batch_norms

from custom_dataset import IndexCIFAR10

#package for computing individual gradients
from backpack import backpack, extend
from backpack.extensions import BatchGrad

from idp_tracker import PrivacyLossTracker

parser = argparse.ArgumentParser(description='Train resnet20 on CIFAR-10 with DP-SGD')

## general arguments
parser.add_argument('--sess', default='resnet20_cifar10', type=str, help='session name')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='weight decay')
parser.add_argument('--batchsize', default=2000, type=int, help='batch size')
parser.add_argument('--n_epoch', default=200, type=int, help='total number of epochs')
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate (default=0.1)')
parser.add_argument('--momentum', default=0.9, type=float, help='value of momentum')


## arguments for learning with differential privacy
parser.add_argument('--private', '-p', action='store_true', help='enable differential privacy')
parser.add_argument('--clip', default=15, type=float, help='gradient upper bound, the larger this value, the more diverse the sample-wise privacy loss')
parser.add_argument('--delta', default=1e-5, type=float, help='desired delta')
parser.add_argument('--sigma', default=1.5, type=float, help='noise multiplier')
parser.add_argument('--rounding', default=0.1, type=float, help='value used for rounding the gradient norm')
parser.add_argument('--update_freq', default=1, type=float, help='how many times in an epoch we update the full gradient norms')

args = parser.parse_args()


use_cuda = True
best_acc = 0  
start_epoch = 0  
batch_size = args.batchsize

# set random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

print('==> Preparing data..')


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])  

trainset = IndexCIFAR10(root='./data', train=True, download=True, transform=transform) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)

testset = IndexCIFAR10(root='./data', train=False, download=True, transform=transform) 
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batchsize, shuffle=False, num_workers=2)



noshuffle_trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=False, num_workers=2)


# Take the first 1000 training samples, we will compute the gradient norms of these samples at every update.
# The norms will be used to compute the exact privacy loss. We compare estimated privacy loss with exact privacy loss to show the accuracy of estimated individual privacy.
_cnt = 0
anchor_norm_samples, anchor_norm_labels = [], []
for (inputs, targets, index) in noshuffle_trainloader:
    x, y = inputs.cuda(), targets.cuda()
    anchor_norm_samples.append(x)
    anchor_norm_labels.append(y)
    _cnt += 1
anchor_norm_samples = torch.cat(anchor_norm_samples)
anchor_norm_labels = torch.cat(anchor_norm_labels)
anchor_norm_samples = anchor_norm_samples[0:1000]
anchor_norm_labels = anchor_norm_labels[0:1000]



n_training = trainset.__len__()
n_test = testset.__len__()

print('# of training examples: ', n_training, '# of testing examples: ', n_test)
sampling_prob=args.batchsize/n_training



steps_per_epoch = (n_training//args.batchsize)
if(n_training%args.batchsize != 0):
    steps_per_epoch += 1

orig_n = n_training

noise_multiplier = args.sigma
print('noise scale: ', noise_multiplier) 



print('\n==> Creating the model..')
net = resnet20()
net.cuda()
for m in net.modules():
    if(hasattr(m, 'inplace')):
        m.inplace = False
# Used by the backpack package to compute per-example gradients
net = extend(net)
num_params = 0
for p in net.parameters():
    num_params += p.numel()


print('total number of parameters: ', num_params/(10**6), 'M')
if(args.private):
    # cross entropy loss
    loss_func = torch.nn.CrossEntropyLoss(reduction='sum')
else:
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean')



loss_func = extend(loss_func)

optimizer = optim.SGD(
        net.parameters(), 
        lr=args.lr, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch)

update_cnts = 0

assert args.update_freq > 0

args.update_per_steps = steps_per_epoch // args.update_freq


# Get the gradient norms of the small subset of training data.
# This is called after every update.
def get_batch_norms(): 
    net.eval()
    inputs, targets = anchor_norm_samples, anchor_norm_labels
    _size = inputs.shape[0]
    _norm_list = []
    steps = _size // args.batchsize
    if _size % args.batchsize != 0:
        steps += 1
    bs = args.batchsize
    for i in range(steps):
        _inputs, _targets = inputs[i*bs:(i+1)*bs], targets[i*bs:(i+1)*bs]
        optimizer.zero_grad()
        outputs = net(_inputs)
        loss = loss_func(outputs, _targets)
        with backpack(BatchGrad()):
            loss.backward()
            norms = get_grad_batch_norms(list(net.named_parameters())) # clip gradients and sum clipped gradients
        _norm_list.append(norms)

    return torch.cat(_norm_list)


# Update the gradient norms of the whole training set.
# Only called occasionally.
def update_norms(epoch):
    print('updating norms at epoch %d'%(epoch))
    net.eval()
    steps = (n_training//args.batchsize)
    if(n_training%args.batchsize != 0):
        steps += 1
    loader = iter(trainloader)
    norms_list = []
    idx_list = []
    for batch_idx in range(steps):
        inputs, targets, index = next(loader)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        minibatch_idx = index

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_func(outputs, targets)
        with backpack(BatchGrad()):
            loss.backward()
            norms = get_grad_batch_norms(list(net.named_parameters()))
        idx_list.append(minibatch_idx)
        norms_list.append(norms)

    idx = torch.cat(idx_list)
    norms = torch.cat(norms_list)
    idp_accoutant.update_norm(norms, idx)
    ghost_idp_accountant.update_norm(norms, idx)

# this is used for computing individual privacy with estimates of gradient norms
idp_accoutant = PrivacyLossTracker(n_training, args.batchsize, noise_multiplier, init_norm=args.clip, delta=args.delta, rounding=args.rounding)
idp_accoutant.update_rdp()

# this is used for computing individual privacy with exact gradient norms (only for the first 1000 samples). the results will be used for computing the estimation error.
ghost_idp_accountant = PrivacyLossTracker(n_training, args.batchsize, noise_multiplier, init_norm=args.clip, delta=args.delta, rounding=args.rounding)
ghost_idp_accountant.update_rdp()

def train(epoch):

    global update_cnts

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()

    steps = (n_training//args.batchsize)
    if(n_training%args.batchsize != 0):
        steps += 1

    loader = iter(trainloader)

    for batch_idx in range(steps):
        
        inputs, targets, index = next(loader)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        if(update_cnts % args.update_per_steps == 0):
            # update full gradient norms
            update_norms(epoch)
            net.train()

        if(args.private):
            

            # compute exact gradient norms of the first 1000 training samples
            anchor_norms = get_batch_norms()
            # update the gradient norms of the first 1000 training samples, so we can get the exact privacy loss
            ghost_idp_accountant.update_norm(anchor_norms, np.arange(anchor_norms.shape[0]))


            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            with backpack(BatchGrad()):
                optimizer.zero_grad()
                loss.backward()
                norms = process_grad_batch(list(net.named_parameters()), args.clip) # clip gradients and average clipped gradients
                ## add noise to gradient
                for name, p in net.named_parameters():
                    if('bn' not in name):
                        grad_noise = torch.normal(0, noise_multiplier*args.clip/args.batchsize, size=p.grad.shape, device=p.grad.device)
                        p.grad.data += grad_noise
                    else:
                        p.grad = None

                # update privacy loss
                idp_accoutant.update_loss()
                ghost_idp_accountant.update_loss()
                
            
        else:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            loss.backward()
            for p in net.parameters():
                if hasattr(p, 'grad_batch'):
                    del p.grad_batch


        optimizer.step()
        update_cnts += 1
        step_loss = loss.item()
        if(args.private):
            step_loss /= args.batchsize


        train_loss += step_loss
        _, predicted = torch.max(outputs.data, 1)
        minibatch_correct = predicted.eq(targets.data).float().cpu()


        total += targets.size(0)
        correct += minibatch_correct.sum()
        acc = 100.*float(correct)/float(total)

    scheduler.step()
    t1 = time.time()
    print('Train loss:%.5f'%(train_loss/(batch_idx+1)), 'time: %d s'%(t1-t0), 'train acc:', acc, end=' ')

    return train_loss/batch_idx, acc




def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = loss_func(outputs, targets)
            step_loss = loss.item()
            if(args.private):
                step_loss /= inputs.shape[0]

            test_loss += step_loss 
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct_idx = predicted.eq(targets.data).cpu()
            correct += correct_idx.sum()


        acc = 100.*float(correct)/float(total)
        print('test loss:%.5f'%(test_loss/(batch_idx+1)), 'test acc:', acc)
        if acc > best_acc:
            best_acc = acc

    return test_loss/batch_idx, acc


print('\n==> Strat training')

full_train_loss_arr = []
full_train_correct_arr = []
for epoch in range(start_epoch, args.n_epoch):

    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)    

    # print privacy loss statistics
    if(args.private):
        up_eps = idp_accoutant.parallel_get_eps()
        avg_norm = idp_accoutant.get_avg_norm()
        print('Epoch ', epoch, 'max_eps: ', np.max(up_eps), 'mean_eps: ', np.mean(up_eps), 'min_eps: ', np.min(up_eps), 'average grad norm: ', avg_norm)

if(args.private):
    up_eps = idp_accoutant.parallel_get_eps()
    ghost_up_eps = ghost_idp_accountant.parallel_get_eps()
    print('max_eps: ', np.max(up_eps), 'min_eps: ', np.min(up_eps))
    os.makedirs('stats', exist_ok=True)
    # save individual privacy parameters
    np.save('stats/%s_privacy_profile.npy'%args.sess, up_eps) # shape (n_training)
    np.save('stats/%s_ghost_privacy_profile.npy'%args.sess, ghost_up_eps) # shape (n_training)