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
import timeit

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
parser.add_argument('--gpu', default='3', help='index of gpus to use')
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
parser.add_argument('--resume', default='./cub200/checkpoint_pretrain_vgg16_bn.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--first_epochs', default=5, type=int, metavar='N',
                    help='number of first stage epochs to run')
parser.add_argument('--students', default='beginners', help='user type')


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
    model_main = models.__dict__['vgg16f_bn'](pretrained=True)
    model_main.classifier[-1] = nn.Linear(model_main.classifier[-1].in_features, num_classes)
    model_main = torch.nn.DataParallel(model_main, device_ids=range(len(args.gpu))).cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_main.module.load_state_dict(checkpoint['state_dict_m'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.students == 'beginners':
        all_correct_student = np.load('./cub200/all_correct_random_te.npy')
        all_predicted_student = np.load('./cub200/all_predicted_random_te.npy')
        all_gt_target_student = np.load('./cub200/all_gt_target_random_te.npy')
    else:
        all_correct_student = np.load('./cub200/all_correct_alexnet_te.npy')
        all_predicted_student = np.load('./cub200/all_predicted_alexnet_te.npy')
        all_gt_target_student = np.load('./cub200/all_gt_target_alexnet_te.npy')


    # generate predicted hardness score
    criterion = nn.CrossEntropyLoss().cuda()
    criterion_f = nn.CrossEntropyLoss(reduce=False).cuda()
    prec1, prec5, all_correct_te, all_predicted_te, all_entropy_te, all_class_dis_te, all_gt_target_te = validate(val_loader, model_main, criterion, criterion_f)

    all_predicted_te = all_predicted_te.astype(int)
    np.save('./cub200/all_correct_te_cls_vgg16.npy', all_correct_te)
    np.save('./cub200/all_predicted_te_cls_vgg16.npy', all_predicted_te)
    np.save('./cub200/all_entropy_te_cls_vgg16.npy', all_entropy_te)
    np.save('./cub200/all_class_dis_te_cls_vgg16.npy', all_class_dis_te)
    np.save('./cub200/all_gt_target_te_cls_vgg16.npy', all_gt_target_te)

    all_correct_teacher = np.load('./cub200/all_correct_te_cls_vgg16.npy')
    all_predicted_teacher = np.load('./cub200/all_predicted_te_cls_vgg16.npy')
    all_entropy_teacher = np.load('./cub200/all_entropy_te_cls_vgg16.npy')
    all_class_dis_teacher = np.load('./cub200/all_class_dis_te_cls_vgg16.npy')
    all_gt_target_teacher = np.load('./cub200/all_gt_target_te_cls_vgg16.npy')

    # in order to model machine teaching, the examples we care about should be those that student network misclassified but teacher network make it
    interested_idx = np.intersect1d(np.where(all_correct_student == 0), np.where(all_correct_teacher == 1))
    predicted_class = all_predicted_student[interested_idx]
    counterfactual_class = all_gt_target_student[interested_idx]

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


    dis_extracted_attributes = np.load('./cub200/Dominik2003IT_dis_extracted_attributes_02.npy')
    all_locations = np.zeros((5794, 30))
    with open('./cub200/CUB200_partLocs_gt_te.txt', 'r') as rf:
        for line in rf.readlines():
            locations = line.strip().split()
            for i_part in range(30):
                all_locations[int(locations[-1]), i_part] = round(float(locations[i_part]))
    picked_locations = all_locations[interested_idx, :]


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
        batch_size=1, num_workers=args.workers)

    print(args.students)
    remaining_mask_size_pool = np.arange(0.01, 1.0, 0.01)
    recall, precision = cf_proposal_extraction(val_hard_loader, val_loader, model_main,
                                                                     picked_list, dis_extracted_attributes,
                                                                     picked_locations, predicted_class,
                                                                     remaining_mask_size_pool)



    print(recall)
    print(precision)




def validate(val_loader, model_main, criterion, criterion_f):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model_main.eval()
    end = time.time()

    all_correct_te = []
    all_predicted_te = []
    all_entropy_te = []
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

        loss = criterion(output, target)

        entropy = -1 * F.softmax(output, dim=1) * F.log_softmax(output, dim=1)
        entropy = torch.sum(entropy, dim=1).data.cpu().numpy()
        all_entropy_te = np.concatenate((all_entropy_te, entropy), axis=0)

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
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    # print(' * Testing Prec@1 {top1.avg:.3f}'.format(top1=top1))
    all_class_dis = all_class_dis[1:, :]
    return top1.avg, top5.avg, all_correct_te, all_predicted_te, all_entropy_te, all_class_dis, all_gt_target


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
    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor_hp(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model._modules['module'].classifier(output)  # travel many fc layers
        confidence_score = F.softmax(output, dim=1)
        confidence_score = torch.max(confidence_score, dim=1)[0]
        return target_activations, confidence_score


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
    # if is_cls == False:
    #     threshold = np.sort(mask.flatten())[-int(0.05*224*224)]
    #     mask[mask < threshold] = 0
    #     mask[mask > 0] = 1
    mask = np.concatenate((mask[:, :, np.newaxis], mask[:, :, np.newaxis], mask[:, :, np.newaxis]), axis=2)
    img = np.uint8(255 * mask * img)
    if is_cls == False:
        if np.sum(com_attributes_positions*mask[:,:,0]) > 0:
            x, y = np.where(com_attributes_positions*mask[:,:,0] == 1)
            for i in range(np.size(x)):
                cv2.circle(img, (y[i], x[i]), 5, (0,255,0),-1)

            x, y = np.where((all_attributes_positions - com_attributes_positions) * mask[:, :, 0] == 1)
            for i in range(np.size(x)):
                cv2.circle(img, (y[i], x[i]), 5, (0,0,255),-1)

    # using dark images

    img_dark = img_dark * 0.4
    img_dark = np.uint8(255 * img_dark)
    img_dark[mask > 0] = img[mask > 0]
    img = img_dark

    return img





def picking_examples(dataloader, c_num, image_size, labels):
    image_bank = np.zeros((c_num, 3, image_size, image_size))
    for i, (input, target, index) in enumerate(dataloader):
        input = input.data.cpu()
        target = target.data.cpu()
        image_bank[target, :, :, :] = input
    image_bank = torch.from_numpy(image_bank)
    return image_bank[labels, :, :, :]


def cf_proposal_extraction(val_loader, val_loader_cf, model_main, imglist, dis_extracted_attributes, part_Locs, predicted_class, remaining_mask_size_pool):


    recall = np.zeros((len(imglist)))
    precision = np.zeros((len(imglist)))

    image_bank = picking_examples(val_loader_cf, c_num=200, image_size=224, labels=np.arange(200))
    all_max_p_on_cf = np.zeros((len(imglist), 4))
    all_max_posterior = np.zeros((len(imglist)))
    total_times = []
    model_main.eval()
    for ii, (input, target, index) in enumerate(val_loader):

        dis_attributes = dis_extracted_attributes[predicted_class[index], target]


        if len(dis_attributes) < 1:
            recall[ii] = float('NaN')
            precision[ii] = float('NaN')
            continue

        input = input.cuda()
        _, features = model_main(input, False)
        distractor_img = image_bank[predicted_class[index], :, :, :].unsqueeze(0)
        distractor_img = distractor_img.float()
        distractor_img = distractor_img.cuda()
        _,  features_cf = model_main(distractor_img, False)

        features = features.squeeze()
        features_cf = features_cf.squeeze()

        max_i = 0
        max_j = 0
        max_p = 0
        max_q = 0
        max_posterior_cf = 0

        start = timeit.default_timer()

        for i in range(features.shape[2]):
            for j in range(features.shape[2]):

                for p in range(features.shape[2]):
                    for q in range(features.shape[2]):
                        features[:, i, j] = features_cf[:, p, q]
                        new_output, _ = model_main(features.unsqueeze(0), True)
                        class_dis = F.softmax(new_output, dim=1)
                        class_dis = class_dis.data.cpu().numpy()
                        class_dis = class_dis.squeeze()
                        if class_dis[predicted_class[index]] > max_posterior_cf:
                            max_posterior_cf = class_dis[predicted_class[index]]
                            max_i = i
                            max_j = j
                            max_p = p
                            max_q = q
        all_max_p_on_cf[ii, :] = np.array([max_i, max_j, max_p, max_q])
        np.save('./cub200/all_max_p_on_cf.npy', all_max_p_on_cf)
        all_max_posterior[ii] = max_posterior_cf
        np.save('./cub200/all_max_posterior.npy', all_max_posterior)
        stop = timeit.default_timer()
        print('Time:', stop - start)
        total_times.append(stop - start)
        print('processing batch', ii)

        cf_heatmap = np.zeros((features.shape[2], features.shape[2]))
        cf_heatmap[max_i, max_j] = 1
        cf_heatmap = cv2.resize(cf_heatmap, (224, 224))
        cf_heatmap[cf_heatmap > 0] = 1
        cf_heatmap[cf_heatmap < 1] = 0

        img = cv2.imread(imglist[index])
        img_X_max = np.size(img, axis=0)
        img_Y_max = np.size(img, axis=1)
        part_Locs_example = part_Locs[index, :]
        part_Locs_example = np.concatenate(
            (np.reshape(part_Locs_example[0::2], (-1, 1)), np.reshape(part_Locs_example[1::2], (-1, 1))), axis=1)
        part_Locs_example[:, 0] = 224.0 * part_Locs_example[:, 0] / img_Y_max
        part_Locs_example[:, 1] = 224.0 * part_Locs_example[:, 1] / img_X_max
        part_Locs_example = np.round(part_Locs_example)
        part_Locs_example = part_Locs_example.astype(int)

        all_attributes_positions = np.zeros((224, 224))
        dis_attributes_positions = np.zeros((224, 224))


        dis_attributes = np.array(dis_attributes)
        # print(dis_attributes)

        part_Locs_example_copy = np.copy(part_Locs_example)
        part_Locs_example_copy = part_Locs_example_copy[~np.all(part_Locs_example_copy == 0, axis=1)]
        all_attributes_positions[part_Locs_example_copy[:, 1], part_Locs_example_copy[:, 0]] = 1

        dis_attributes_positions[part_Locs_example[dis_attributes, 1], part_Locs_example[dis_attributes, 0]] = 1
        dis_attributes_positions[0, 0] = 0
        # print(np.sum(dis_attributes_positions))

        cur_recall = np.sum(cf_heatmap * dis_attributes_positions) / np.sum(dis_attributes_positions)

        cur_precision = np.sum(cf_heatmap * dis_attributes_positions) / np.sum(cf_heatmap * all_attributes_positions)

        recall[ii] = cur_recall
        precision[ii] = cur_precision

        print(np.nanmean(recall[:ii], axis=0))
        print(np.nanmean(precision[:ii], axis=0))
    total_times = np.array(total_times)
    np.save('./cub200/exhaustive_vgg_total_times.npy', total_times)
    return np.nanmean(recall, axis=0), np.nanmean(precision, axis=0)



if __name__ == '__main__':
    main()



