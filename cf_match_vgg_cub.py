import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np
import datasets
import models as models
import matplotlib.pyplot as plt
import torchvision.models as torch_models
from extra_setting import *
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import utils
import scipy.io as sio
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import cv2
import seaborn as sns
import operator



model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch end2end cub200 Training')
parser.add_argument('-d', '--dataset', default='cub200', help='dataset name')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet20',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet20)')
parser.add_argument('-c', '--channel', type=int, default=16,
                    help='first conv channel (default: 16)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--gpu', default='4', help='index of gpus to use')
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lr_step', default='5', help='decreasing strategy')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./cub200/checkpoint_vgg16bn_hp.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=5, type=int, metavar='N',
                    help='number of first stage epochs to run')


def main():
    global args, best_prec1
    args = parser.parse_args()

    # select gpus
    args.gpu = args.gpu.split(',')
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.gpu)

    # data loader
    assert callable(datasets.__dict__[args.dataset])
    get_dataset = getattr(datasets, args.dataset)
    num_classes = datasets._NUM_CLASSES[args.dataset]
    train_loader, val_loader = get_dataset(
        batch_size=args.batch_size, num_workers=args.workers)

    # create model
    model_main = models.__dict__['vgg16_bn'](pretrained=True)
    model_main.classifier[-1] = nn.Linear(model_main.classifier[-1].in_features, num_classes)
    model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_main.module.load_state_dict(checkpoint['state_dict_m'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model_ahp_trunk = models.__dict__['vgg16_bn'](pretrained=True)
    model_ahp_trunk.classifier[-1] = nn.Linear(model_ahp_trunk.classifier[-1].in_features, 1000)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_ahp_trunk.load_state_dict(checkpoint['state_dict_ahp_trunk'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model_ahp_trunk = torch.nn.DataParallel(model_ahp_trunk, device_ids=range(len(args.gpu))).cuda()

    model_ahp_hp = models.__dict__['ahp_net_hp_res50_presigmoid']()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_ahp_hp.load_state_dict(checkpoint['state_dict_ahp_hp'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model_ahp_hp = torch.nn.DataParallel(model_ahp_hp, device_ids=range(len(args.gpu))).cuda()


    all_correct_student = np.load('./cub200/all_correct_alexnet_te.npy')
    all_predicted_student = np.load('./cub200/all_predicted_alexnet_te.npy')
    all_class_dis_student = np.load('./cub200/all_class_dis_alexnet_te.npy')
    all_gt_target_student = np.load('./cub200/all_gt_target_alexnet_te.npy')

    # generate predicted hardness score
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_f = nn.CrossEntropyLoss(reduce=False).cuda()
    prec1, prec5, all_correct_te, all_predicted_te, all_class_dis_te, all_gt_target_te = validate(val_loader, model_main, model_ahp_trunk,
                                                               model_ahp_hp, criterion, criterion_f)
    all_predicted_te = all_predicted_te.astype(int)
    np.save('./cub200/all_correct_vgg16_te.npy', all_correct_te)
    np.save('./cub200/all_predicted_vgg16_te.npy', all_predicted_te)
    np.save('./cub200/all_class_dis_vgg16_te.npy', all_class_dis_te)
    np.save('./cub200/all_gt_target_vgg16_te.npy', all_gt_target_te)

    all_correct_teacher = np.load('./cub200/all_correct_vgg16_te.npy')
    all_predicted_teacher = np.load('./cub200/all_predicted_vgg16_te.npy')
    all_class_dis_teacher = np.load('./cub200/all_class_dis_vgg16_te.npy')
    all_gt_target_teacher = np.load('./cub200/all_gt_target_vgg16_te.npy')


    # in order to model machine teaching, the examples we care about should be those that student network misclassified but teacher network make it
    interested_idx = np.intersect1d(np.where(all_correct_student == 0), np.where(all_correct_teacher == 1))
    predicted_class = all_predicted_student[interested_idx]
    counterfactual_class = all_gt_target_student[interested_idx]

    # find all cross match, i.e. 0 1 and 1 0 both exist in predicted_class and counterfactual_class
    cross_match1 = np.concatenate((np.expand_dims(predicted_class, axis=1), np.expand_dims(counterfactual_class, axis=1)), axis=1)
    cross_match1 = cross_match1.tolist()
    cross_match2 = np.concatenate((np.expand_dims(counterfactual_class, axis=1), np.expand_dims(predicted_class, axis=1)), axis=1)
    cross_match2 = cross_match2.tolist()
    remain_idx = []
    for i in np.arange(len(cross_match1)):
        if cross_match1[i] in cross_match2:
            remain_idx.append(i)

    interested_idx = interested_idx[remain_idx]
    predicted_class = predicted_class[remain_idx]
    counterfactual_class = counterfactual_class[remain_idx]
    counterfactual_class.astype(int)
    cross_match1 = np.array(cross_match1)
    cross_match1.astype(int)
    cross_match1 = cross_match1[remain_idx]
    cross_match1[:, [0, 1]] = cross_match1[:, [1, 0]]
    cross_match, indices = np.unique(cross_match1, axis=0, return_index=True)
    interested_idx = interested_idx[indices]
    predicted_class = predicted_class[indices]
    counterfactual_class = counterfactual_class[indices]

    cross_match[:, [0, 1]] = cross_match[:, [1, 0]]

    # pick the interested images
    imlist = []
    imclass = []
    with open('./cub200/CUB_200_2011/CUB200_gt_te.txt', 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, imindex = line.strip().split()
            imlist.append(impath)
            imclass.append(imlabel)

    picked_list = []
    picked_class_list = []
    for i in range(np.size(interested_idx)):
        picked_list.append(imlist[interested_idx[i]])
        picked_class_list.append(imclass[interested_idx[i]])

    heat_map_hp = Heatmap_hp(model_ahp_trunk, model_ahp_hp, target_layer_names=["42"], use_cuda=True)
    heat_map_cls = Heatmap_cls(model_main, target_layer_names=["42"], use_cuda=True)

    dis_extracted_attributes = np.load('./cub200/Dominik2003IT_dis_extracted_attributes_02.npy')
    all_locations = np.zeros((5794, 30))
    with open('./cub200/CUB200_partLocs_gt_te.txt', 'r') as rf:
        for line in rf.readlines():
            locations = line.strip().split()
            for i_part in range(30):
                all_locations[int(locations[-1]), i_part] = round(float(locations[i_part]))
    picked_locations = all_locations[interested_idx, :]



    # save cub200 hard info
    cub200cf = './cub200/CUB200cf_gt_te.txt'
    fl = open(cub200cf, 'w')
    num_cf = 0
    for ii in range(len(picked_list)):

        example_info = picked_list[ii] + " " + picked_class_list[ii] + " " + str(num_cf)
        fl.write(example_info)
        fl.write("\n")
        num_cf = num_cf + 1
    fl.close()

    # data loader
    assert callable(datasets.__dict__['cub200cf'])
    get_dataset = getattr(datasets, 'cub200cf')
    num_classes = datasets._NUM_CLASSES['cub200cf']
    _, val_hard_loader = get_dataset(
        batch_size=25, num_workers=args.workers)

    train_loader_for_show = './cub200/CUB_200_2011/CUB200_gt_tr.txt'
    remaining_mask_size_pool = np.arange(0.01, 0.2, 0.01)
    cf_proposal_extraction(train_loader_for_show, val_hard_loader, heat_map_hp, heat_map_cls,
                                                                     picked_list, dis_extracted_attributes,
                                                                     picked_locations, predicted_class,
                                                                     remaining_mask_size_pool, cross_match)






def validate(val_loader, model_main, model_ahp_trunk, model_ahp_hp, criterion, criterion_f):
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_main.eval()
    model_ahp_trunk.eval()
    model_ahp_hp.eval()
    end = time.time()

    all_correct_te = []
    all_predicted_te = []
    all_class_dis = np.zeros((1, 200))
    all_gt_target = []
    for i, (input, target, index) in enumerate(val_loader):

        all_gt_target = np.concatenate((all_gt_target, target), axis=0)

        input = input.cuda()
        target = target.cuda(async=True)

        # compute output
        output = model_main(input)
        class_dis = F.softmax(output, dim=1)
        class_dis = class_dis.data.cpu().numpy()
        all_class_dis = np.concatenate((all_class_dis, class_dis), axis=0)


        p_i_m = torch.max(output, dim=1)[1]
        all_predicted_te = np.concatenate((all_predicted_te, p_i_m), axis=0)
        p_i_m = p_i_m.long()
        p_i_m[p_i_m - target == 0] = -1
        p_i_m[p_i_m > -1] = 0
        p_i_m[p_i_m == -1] = 1
        correct = p_i_m.float()
        all_correct_te = np.concatenate((all_correct_te, correct), axis=0)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time,
                top1=top1, top5=top5))

    all_class_dis = all_class_dis[1:, :]
    return top1.avg, top5.avg, all_correct_te, all_predicted_te, all_class_dis, all_gt_target


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def largest_indices_each_example(all_response, topK):
    topK_maxIndex = np.zeros((np.size(all_response, 0), topK), dtype=np.int16)
    topK_maxValue = np.zeros((np.size(all_response, 0), topK))
    for i in range(np.size(topK_maxIndex, 0)):
        arr = all_response[i, :]
        topK_maxIndex[i, :] = np.argsort(arr)[-topK:][::-1]
        topK_maxValue[i, :] = np.sort(arr)[-topK:][::-1]
    return topK_maxIndex, topK_maxValue



def save_checkpoint(state, filename='checkpoint_res.pth.tar'):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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





class FeatureExtractor_hp():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)  # forward one layer each time
            if name in self.target_layers:  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows

        x = x.view(x.size(0), -1)
        x = self.model._modules['module'].classifier(x)
        return outputs, x



class FeatureExtractor_cls():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules['module']._modules['features']._modules.items():
            x = module(x)  # forward one layer each time
            if name in self.target_layers:  # store the gradient of target layer
                x.register_hook(self.save_gradient)
                outputs += [x]  # after last feature map, nn.MaxPool2d(kernel_size=2, stride=2)] follows
        return outputs, x

class ModelOutputs_hp():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model_hp_trunk, model_hp_head, target_layers):
        self.model_hp_trunk = model_hp_trunk
        self.model_hp_head = model_hp_head
        self.feature_extractor = FeatureExtractor_hp(self.model_hp_trunk, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output, _ = self.model_hp_head(output)
        return target_activations, output


class ModelOutputs_cls():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_cls(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model._modules['module'].classifier(output)  # travel many fc layers
        return target_activations, output


def preprocess_image(img):
    # means=[0.485, 0.456, 0.406]
    # stds=[0.229, 0.224, 0.225]
    means = [0.4706145, 0.46000465, 0.45479808]
    stds = [0.26668432, 0.26578658, 0.2706199]

    preprocessed_img = img.copy()[: , :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad = True)
    return input

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def show_segment_on_image(img, mask, com_attributes_positions=None, all_attributes_positions=None, is_cls=True):
    img = np.float32(img)
    img_dark = np.copy(img)
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * mask * img)
    if is_cls == False:
        if np.sum(com_attributes_positions*mask[:,:,0]) > 0:
            x, y = np.where(com_attributes_positions*mask[:,:,0] == 1)
            for i in range(np.size(x)):
                cv2.circle(img, (y[i], x[i]), 2, (0,255,0),-1)

    img_dark = img_dark * 0.4
    img_dark = np.uint8(255 * img_dark)
    img_dark[mask > 0] = img[mask > 0]
    img = img_dark

    return img


def show_segment_on_image3(img, mask, com_attributes_positions=None):

    # draw contours for masked regions

    img = np.float32(img)
    img_dark = np.copy(img)
    mask255 = np.copy(mask)
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * img)
    if np.sum(com_attributes_positions * mask[:, :, 0]) > 0:
        x, y = np.where(com_attributes_positions * mask[:, :, 0] == 1)
        for i in range(np.size(x)):
            cv2.circle(img, (y[i], x[i]), 2, (0, 255, 0), -1)


    mask255 = np.uint8(255 * mask255)
    contours, hierarchy = cv2.findContours(mask255, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # img_dark = img_dark * 0.4
    # img_dark = np.uint8(255 * img_dark)
    # img_dark[mask > 0] = img[mask > 0]
    # img = img_dark

    return img


def show_segment_on_image2(img, mask, com_attributes_positions=None, all_attributes_positions=None, is_cls=True):
    # show all positive and negative

    img = np.float32(img)
    img_dark = np.copy(img)
    # if is_cls == False:
    #     threshold = np.sort(mask.flatten())[-int(0.05*224*224)]
    #     mask[mask < threshold] = 0
    #     mask[mask > 0] = 1
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * img)
    if is_cls == False:
        x, y = np.where(com_attributes_positions == 1)
        for i in range(np.size(x)):
            cv2.circle(img, (y[i], x[i]), 5, (0,255,0),-1)

        x, y = np.where((all_attributes_positions - com_attributes_positions) == 1)
        for i in range(np.size(x)):
            cv2.circle(img, (y[i], x[i]), 5, (0,0,255),-1)

    # using dark images
    img_dark = img * 0.4

    img_dark[mask > 0] = img[mask > 0]
    img = img_dark

    return img





class Heatmap_hp:
    def __init__(self, model_hp_trunk, model_hp_head, target_layer_names, use_cuda):
        self.model_hp_trunk = model_hp_trunk
        self.model_hp_head = model_hp_head
        self.model_hp_trunk.eval()
        self.model_hp_head.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model_hp_trunk = model_hp_trunk.cuda()
            self.model_hp_head = model_hp_head.cuda()

        self.extractor = ModelOutputs_hp(self.model_hp_trunk, self.model_hp_head, target_layer_names)

    def forward(self, input):
        return self.model_hp_head(self.model_hp_trunk(input))

    def __call__(self, input):

        features, output = self.extractor(input)
        output = 1.0 - output  # get the easiness scores

        grads_val = torch.autograd.grad(output, features[0], grad_outputs=torch.ones_like(output),
                                           create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy()

        mask_positive = np.copy(grads_val)
        mask_positive[mask_positive < 0.0] = 0.0
        mask_positive = mask_positive.squeeze()

        target = features[-1]
        target = target.cpu().data.numpy()

        cam_positive = target * mask_positive
        cam_positive = np.sum(cam_positive, axis=1)
        return cam_positive


class Heatmap_cls:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs_cls(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, PredictedClass, CounterfactualClass):

        features, output = self.extractor(input)

        target = features[-1]
        target = target.cpu().data.numpy()

        classifier_heatmaps = np.zeros((input.shape[0], np.size(target, 2), np.size(target, 2), 2))
        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(output.shape[0]), PredictedClass] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        grads_val = torch.autograd.grad(one_hot, features, grad_outputs=torch.ones_like(one_hot),
                                        create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy().squeeze()

        cam_positive = target * grads_val
        cam_positive = np.sum(cam_positive, axis=1)
        classifier_heatmaps[:, :, :, 0] = cam_positive

        one_hot = np.zeros((output.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(output.shape[0]), CounterfactualClass] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = torch.sum(one_hot.cuda() * output, dim=1)
        grads_val = torch.autograd.grad(one_hot, features, grad_outputs=torch.ones_like(one_hot),
                                        create_graph=True)
        grads_val = grads_val[0].squeeze()
        grads_val = grads_val.cpu().data.numpy().squeeze()

        cam_positive = target * grads_val
        cam_positive = np.sum(cam_positive, axis=1)
        classifier_heatmaps[:, :, :, 1] = cam_positive


        return classifier_heatmaps


def picking_examples(train_list, c_num, image_size, labels):
    image_bank = np.zeros((c_num, image_size, image_size, 3))
    indicator_vector = np.zeros((c_num))
    with open(train_list, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel, _ = line.strip().split()
            imlabel = int(imlabel)
            if indicator_vector[imlabel] == 0:
                img = cv2.imread(impath)
                img = np.float32(cv2.resize(img, (224, 224))) / 255
                img = np.uint8(255 * img)
                image_bank[imlabel, :, :, :] = img
                indicator_vector[imlabel] = 1

    return image_bank[labels, :, :, :]


def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def part_included_calculation(part_Locs_example, cf_heatmap):
    cur_included_part = np.zeros((15))
    for i in range(15):
        all_attributes_positions = np.zeros((224, 224))
        all_attributes_positions[part_Locs_example[i, 1], part_Locs_example[i, 0]] = 1
        all_attributes_positions[0, 0] = 0
        if np.sum(all_attributes_positions*cf_heatmap) > 0:
            cur_included_part[i] = 1
    return cur_included_part

def positive_part_included_calculation(part_Locs_example, cf_heatmap, dis_attributes):
    cur_included_part = np.zeros((15))
    for i in range(15):
        all_attributes_positions = np.zeros((224, 224))
        all_attributes_positions[part_Locs_example[i, 1], part_Locs_example[i, 0]] = 1
        all_attributes_positions[0, 0] = 0
        if np.sum(all_attributes_positions*cf_heatmap) > 0:
            cur_included_part[i] = 1
    # to remain positive points
    cur_included_part[dis_attributes] = cur_included_part[dis_attributes] + 1
    cur_included_part[cur_included_part < 2] = 0
    cur_included_part[cur_included_part > 1] = 1
    return cur_included_part

def part_matching(query_img, distractor_img, white_rectangle, positive_part_common, part_Locs_example_query, part_Locs_example_distractor):
    concatenated_image = np.concatenate((query_img, white_rectangle, distractor_img), axis=1)
    parts = np.where(positive_part_common > 0)
    parts = parts[0]
    for i in range(np.size(parts)):
        x_q = part_Locs_example_query[parts[i], 1].squeeze()
        y_q = part_Locs_example_query[parts[i], 0].squeeze()
        x_d = part_Locs_example_distractor[parts[i], 1].squeeze()
        y_d = part_Locs_example_distractor[parts[i], 0].squeeze()+224+np.size(white_rectangle, 1)

        cv2.line(concatenated_image, (y_q, x_q), (y_d, x_d), (0, 200, 20), 1)

    return concatenated_image


def cf_proposal_extraction(train_loader, val_loader, heat_map_hp, heat_map_cls, imglist, dis_extracted_attributes, part_Locs, predicted_class, remaining_mask_size_pool, cross_match):

    all_region_proposals = np.zeros((len(imglist), np.size(remaining_mask_size_pool), 224, 224, 3))
    included_part = np.zeros((len(imglist), np.size(remaining_mask_size_pool), 15))
    all_images = np.zeros((len(imglist), 224, 224, 3))
    included_positive_part = np.zeros((len(imglist), np.size(remaining_mask_size_pool), 15))
    all_heatmap_mask = np.zeros((len(imglist), np.size(remaining_mask_size_pool), 224, 224))

    all_X_Y_max = np.zeros((len(imglist), 2))

    i_sample = 0
    for i, (input, target, index) in enumerate(val_loader):

        input = input.cuda()

        print('processing batch', i)

        easiness_heatmaps_set = heat_map_hp(input)
        easiness_mask_set = np.copy(easiness_heatmaps_set)
        easiness_mask_set[easiness_mask_set > 0] = 1
        classifier_heatmaps_set = heat_map_cls(input, predicted_class[index], target)
        classifier_heatmaps_set[classifier_heatmaps_set < 0] = 1e-7
        predicted_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 0]
        counterfactual_class_heatmaps_set = classifier_heatmaps_set[:, :, :, 1]

        for i_batch in range(index.shape[0]):
            easiness_heatmaps = easiness_heatmaps_set[i_batch, :, :].squeeze()
            easiness_mask = easiness_mask_set[i_batch, :, :].squeeze()
            predicted_class_heatmaps = predicted_class_heatmaps_set[i_batch, :, :].squeeze()
            counterfactual_class_heatmaps = counterfactual_class_heatmaps_set[i_batch, :, :].squeeze()

            img = cv2.imread(imglist[index[i_batch]])
            img_X_max = np.size(img, axis=0)
            img_Y_max = np.size(img, axis=1)
            img = np.float32(cv2.resize(img, (224, 224))) / 255
            part_Locs_example = part_Locs[index[i_batch], :]
            part_Locs_example = np.concatenate((np.reshape(part_Locs_example[0::2], (-1, 1)), np.reshape(part_Locs_example[1::2], (-1, 1))), axis=1)
            part_Locs_example[:, 0] = 224.0 * part_Locs_example[:, 0] / img_Y_max
            part_Locs_example[:, 1] = 224.0 * part_Locs_example[:, 1] / img_X_max
            part_Locs_example = np.round(part_Locs_example)
            part_Locs_example = part_Locs_example.astype(int)

            all_X_Y_max[index[i_batch], :] = img_X_max, img_Y_max

            all_images[i_sample, :, :, :] = img

            for i_remain in range(np.size(remaining_mask_size_pool)):
                remaining_mask_size = remaining_mask_size_pool[i_remain]

                cf_heatmap = easiness_heatmaps * (np.amax(predicted_class_heatmaps) - predicted_class_heatmaps) * counterfactual_class_heatmaps
                cf_heatmap = cv2.resize(cf_heatmap, (224, 224))
                threshold = np.sort(cf_heatmap.flatten())[int(-remaining_mask_size * 224 * 224)]
                cf_heatmap[cf_heatmap > threshold] = 1
                cf_heatmap[cf_heatmap < 1] = 0

                all_attributes_positions = np.zeros((224, 224))
                dis_attributes_positions = np.zeros((224, 224))

                dis_attributes = dis_extracted_attributes[predicted_class[index[i_batch]], target[i_batch]]

                if len(dis_attributes) < 1:
                    continue

                dis_attributes = np.array(dis_attributes)

                part_Locs_example_copy = np.copy(part_Locs_example)
                part_Locs_example_copy = part_Locs_example_copy[~np.all(part_Locs_example_copy == 0, axis=1)]
                all_attributes_positions[part_Locs_example_copy[:, 1], part_Locs_example_copy[:, 0]] = 1

                dis_attributes_positions[part_Locs_example[dis_attributes, 1], part_Locs_example[dis_attributes, 0]] = 1
                dis_attributes_positions[0, 0] = 0

                included_part[i_sample, i_remain, :] = part_included_calculation(part_Locs_example, cf_heatmap)
                included_positive_part[i_sample, i_remain, :] = positive_part_included_calculation(part_Locs_example, cf_heatmap, dis_attributes)

                all_heatmap_mask[i_sample, i_remain, :, :] = cf_heatmap

                seg = show_segment_on_image(img, cf_heatmap, dis_attributes_positions, all_attributes_positions,
                                            is_cls=False)
                all_region_proposals[i_sample, i_remain, :, :, :] = seg
            i_sample = i_sample + 1

    # plot and save
    predicted_class = cross_match[:, 0]
    counterfactual_class = cross_match[:, 1]

    match_points_IOU = np.zeros((len(imglist), np.size(remaining_mask_size_pool)))

    for i_remain in range(np.size(remaining_mask_size_pool)):
        remaining_mask_size = remaining_mask_size_pool[i_remain]

        for i in range(len(imglist)):

            query_img_example = all_images[i, :, :, :]

            distractor_img_idx = np.intersect1d(np.where(predicted_class == counterfactual_class[i]),
                                                np.where(counterfactual_class == predicted_class[i]))
            distractor_img_idx = distractor_img_idx[0]
            distractor_img_example = all_images[distractor_img_idx, :, :, :]

            positive_part_query = included_positive_part[i, i_remain, :]
            positive_part_distractor = included_positive_part[distractor_img_idx, i_remain, :]
            positive_part_common = positive_part_query * positive_part_distractor

            if np.sum(positive_part_common) > 1:

                dis_attributes_positions = np.zeros((224, 224))
                dis_attributes = np.where(positive_part_common > 1)
                dis_attributes = dis_attributes[0]

                part_Locs_example = part_Locs[i, :]
                part_Locs_example = np.concatenate((np.reshape(part_Locs_example[0::2], (-1, 1)), np.reshape(part_Locs_example[1::2], (-1, 1))), axis=1)
                part_Locs_example[:, 0] = 224.0 * part_Locs_example[:, 0] / all_X_Y_max[i, 1]
                part_Locs_example[:, 1] = 224.0 * part_Locs_example[:, 1] / all_X_Y_max[i, 0]
                part_Locs_example = np.round(part_Locs_example)
                part_Locs_example_query = part_Locs_example.astype(int)


                dis_attributes_positions[part_Locs_example_query[dis_attributes, 1], part_Locs_example_query[dis_attributes, 0]] = 1
                dis_attributes_positions[0, 0] = 0

                query_img = show_segment_on_image3(query_img_example, all_heatmap_mask[i, i_remain, :, :].squeeze(), dis_attributes_positions)

                dis_attributes_positions = np.zeros((224, 224))

                part_Locs_example = part_Locs[distractor_img_idx, :]
                part_Locs_example = np.concatenate(
                    (np.reshape(part_Locs_example[0::2], (-1, 1)), np.reshape(part_Locs_example[1::2], (-1, 1))),
                    axis=1)
                part_Locs_example[:, 0] = 224.0 * part_Locs_example[:, 0] / all_X_Y_max[distractor_img_idx, 1]
                part_Locs_example[:, 1] = 224.0 * part_Locs_example[:, 1] / all_X_Y_max[distractor_img_idx, 0]
                part_Locs_example = np.round(part_Locs_example)
                part_Locs_example_distractor = part_Locs_example.astype(int)

                dis_attributes_positions[part_Locs_example_distractor[dis_attributes, 1], part_Locs_example_distractor[dis_attributes, 0]] = 1
                dis_attributes_positions[0, 0] = 0

                distractor_img = show_segment_on_image3(distractor_img_example, all_heatmap_mask[distractor_img_idx, i_remain, :, :].squeeze(), dis_attributes_positions)

                query_img_example = np.float32(query_img_example)
                query_img_example = np.uint8(255 * query_img_example)
                distractor_img_example = np.float32(distractor_img_example)
                distractor_img_example = np.uint8(255 * distractor_img_example)

                white_rectangle = np.ones((224, 10, 3), dtype=np.float32)
                white_rectangle = np.uint8(255 * white_rectangle)

                matched_query_distractor = part_matching(query_img, distractor_img, white_rectangle, positive_part_common, part_Locs_example_query, part_Locs_example_distractor)

                img_save = np.concatenate((query_img, white_rectangle, distractor_img), axis=1)



                if not os.path.exists("./cub200/cf_regions_vgg/" + str(i_remain)):
                    os.makedirs("./cub200/cf_regions_vgg/" + str(i_remain))

                cv2.imwrite("./cub200/cf_regions_vgg/" + str(i_remain) + "/" + str(remaining_mask_size) + "_" + str(
                    counterfactual_class[i]) + "_" + str(predicted_class[i]) + ".png", img_save)


            query_img_points = included_part[i, i_remain, :]
            distractor_img_points = included_part[distractor_img_idx, i_remain, :]

            match_points_IOU[i, i_remain] = np.sum(query_img_points*distractor_img_points) / (np.sum(query_img_points) + np.sum(distractor_img_points) - np.sum(query_img_points*distractor_img_points))


if __name__ == '__main__':
    main()



