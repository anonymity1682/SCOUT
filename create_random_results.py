import numpy as np
import random


# on cub200
imlist = []
imclass = []

with open('./cub200/CUB_200_2011/CUB200_gt_te.txt', 'r') as rf:
    for line in rf.readlines():
        impath, imlabel, imindex = line.strip().split()
        imlist.append(impath)

        imclass.append(int(imlabel))


all_gt_target_te = np.array(imclass)
all_predicted_te = np.random.randint(200, size=len(imlist))

all_correct_te = all_gt_target_te == all_predicted_te
all_correct_te = all_correct_te * 1  # boolean to int

np.save('./cub200/all_correct_random_te.npy', all_correct_te)
np.save('./cub200/all_predicted_random_te.npy', all_predicted_te)
np.save('./cub200/all_gt_target_random_te.npy', all_gt_target_te)


# on ade
imlist = []
imclass = []

with open('./ade/ADEChallengeData2016/ADE_gt_val.txt', 'r') as rf:
    for line in rf.readlines():
        impath, imlabel, imindex = line.strip().split()
        imlist.append(impath)

        imclass.append(int(imlabel))


all_gt_target_te = np.array(imclass)
all_predicted_te = np.random.randint(1040, size=len(imlist))

all_correct_te = all_gt_target_te == all_predicted_te
all_correct_te = all_correct_te * 1  # boolean to int

np.save('./ade/all_correct_random_te.npy', all_correct_te)
np.save('./ade/all_predicted_random_te.npy', all_predicted_te)
np.save('./ade/all_gt_target_random_te.npy', all_gt_target_te)