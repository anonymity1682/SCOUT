# SCOUT

![scout](/Figs/cf.PNG)


## Requirements

1. The project was implemented and tested in Python 3.5 and Pytorch 0.4. The higher versions should work after minor modification.
2. Other common modules like numpy, pandas and seaborn for visualization.
3. NVIDIA GPU and cuDNN are required to have fast speeds. For now, CUDA 8.0 with cuDNN 6.0.20 has been tested. The other versions should be working.


## Datasets

[CUB200](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and [ADE20K](http://sceneparsing.csail.mit.edu/) are used. Please organize them as below after download,


```
cub200
|_ CUB_200_2011
  |_ attributes
  |_ images
  |_ parts
  |_ train_test_split.txt
  |_ ...
```

```
ade
|_ ADEChallengeData2016
  |_ annotations
  |_ images
  |_ objectInfo 150.txt
  |_ sceneCategories.txt
```

## Implementation details

### data preparation

build train/validation/test sets,

```
make_cub_list.py
make_ade_list.py
```

compute ground truth parts on CUB200 and objects on ADE20K,

```
make_gt_cub.py
make_gt_ade.py
```

prepare attribute location data on CUB200

```
get_gt_partLocs.py
```

### training

Two types of models need to be trained, the standard CNN classifier and [Hardness predictor](http://openaccess.thecvf.com/content_ECCV_2018/html/Pei_Wang_Towards_Realistic_Predictors_ECCV_2018_paper.html). Three most popular architectures were tested. For reproducing each result individually, we separately wrote the code for each experiment. For the classifier,
```
./cub200/train_cub_alexnet.py
./cub200/train_cub_vgg.py
./cub200/train_cub_res.py
./ade/train_ade_alexnet.py
./ade/train_ade_vgg.py
./ade/train_ade_res.py
```
for the hardness predictor,
```
./cub200/train_hp_cub_vgg.py
./cub200/train_hp_cub_res.py
./ade/train_hp_ade_vgg.py
./ade/train_hp_ade_res.py
```

### visualization

1. To reproduce results on sec 5.1
```
python cf_ss_vgg_cub.py --student=beginners --maps=a
python cf_ss_vgg_cub.py --student=beginners --maps=ab
python cf_ss_vgg_cub.py --student=beginners --maps=abs
python cf_cs_vgg_cub.py --student=beginners --maps=abs
python cf_es_vgg_cub.py --student=beginners --maps=abs
python cf_ss_vgg_cub.py --student=advanced --maps=a
python cf_ss_vgg_cub.py --student=advanced --maps=ab
python cf_ss_vgg_cub.py --student=advanced --maps=abs
python cf_cs_vgg_cub.py --student=advanced --maps=abs
python cf_es_vgg_cub.py --student=advanced --maps=abs
```

2. To reproduce results on sec 5.2
```
python cf_ss_vgg_cub.py --student=beginners
python cf_ss_vgg_cub.py --student=advanced
python cf_ss_res_cub.py --student=beginners
python cf_ss_res_cub.py --student=advanced
python cf_PIoU_ss_vgg_cub.py --student=beginners
python cf_PIoU_ss_vgg_cub.py --student=advanced
python cf_exhaustive_vgg_cub.py --student=beginners
python cf_exhaustive_vgg_cub.py --student=advanced
python cf_exhaustive_res_cub.py --student=beginners
python cf_exhaustive_res_cub.py --student=advanced
python cf_PIoU_exhaustive_vgg_cub.py --student=beginners
python cf_PIoU_exhaustive_vgg_cub.py --student=advanced
```

3. To reproduce results on sec 5.3
```
python cf_match_res_cub.py
python cf_match_exhaustive_cub.py
python cf_match_vgg_ade.py
```

4. To produce results on sec 1 of supplement
```
python cf_ss_vgg_ade.py --student=beginners --maps=a
python cf_ss_vgg_ade.py --student=beginners --maps=ab
python cf_ss_vgg_ade.py --student=beginners --maps=abs
python cf_cs_vgg_ade.py --student=beginners --maps=abs
python cf_es_vgg_ade.py --student=beginners --maps=abs
python cf_ss_vgg_ade.py --student=advanced --maps=a
python cf_ss_vgg_ade.py --student=advanced --maps=ab
python cf_ss_vgg_ade.py --student=advanced --maps=abs
python cf_cs_vgg_ade.py --student=advanced --maps=abs
python cf_es_vgg_ade.py --student=advanced --maps=abs
```

5. To reproduce results on sec 2 of supplement
```
python cf_ss_vgg_unique_cub.py --student=beginners --maps=a
python cf_ss_vgg_unique_cub.py --student=beginners --maps=as
python cf_cs_vgg_unique_cub.py --student=beginners --maps=as
python cf_es_vgg_unique_cub.py --student=beginners --maps=as
python cf_ss_vgg_unique_cub.py --student=advanced --maps=a
python cf_ss_vgg_unique_cub.py --student=advanced --maps=as
python cf_cs_vgg_unique_cub.py --student=advanced --maps=as
python cf_es_vgg_unique_cub.py --student=advanced --maps=as
python cf_ss_vgg_unique_ade.py --student=beginners --maps=a
python cf_ss_vgg_unique_ade.py --student=beginners --maps=as
python cf_cs_vgg_unique_ade.py --student=beginners --maps=as
python cf_es_vgg_unique_ade.py --student=beginners --maps=as
python cf_ss_vgg_unique_ade.py --student=advanced --maps=a
python cf_ss_vgg_unique_ade.py --student=advanced --maps=as
python cf_cs_vgg_unique_ade.py --student=advanced --maps=as
python cf_es_vgg_unique_ade.py --student=advanced --maps=as
```

6. To reproduce results on sec 3 of supplement. Three types of attribution methods are compared, baseline [gradient based](https://arxiv.org/abs/1312.6034), [Grad-CAM](https://ieeexplore.ieee.org/document/8237336), state-of-the-art [integrated gradient (IG) based](https://dl.acm.org/citation.cfm?id=3306024).
```
python cf_es_vgg_cub.py
python cf_es_res34_cub.py
python cf_es_res50_cub.py
python cf_es_vgg_ade.py
python cf_es_res34_ade.py
python cf_es_res50_ade.py
python cf_es_vgg_cub_IG.py
python cf_es_vgg_cub_gradcam.py
python cf_es_vgg_ade_IG.py
python cf_es_vgg_ade_gradcam.py
```


### pretrained models

The [pre-trained models](https://drive.google.com/drive/folders/1fh1HMqjrFFctkTgjYvylEiUhEQP3aZOg?usp=sharing) for all experiments are availiable.




## Time and Space

All experiments were run on NVIDIA TITAN Xp 

### training

1. CUB200 (Train/Val/Test Size: 5395/599/5794, please refer to our paper for other setting details.)


model     | #GPUs | train time |
---------|--------|-----|
AlexNet-CNN-baseline     | 1 | ~50min    | 
VGG16-CNN-baseline     | 2 | ~70min    |
Res50-CNN-baseline     | 1 | ~60min    | 
VGG16-HardnessPredictor     | 4 | ~120min   |
Res50-HardnessPredictor     | 2 | ~100min    |

2. ADE20K (Train/Val/Test Size: 18189/2021/2000, please refer to our paper for other setting details.)


model     | #GPUs | train time |
---------|--------|-----|
AlexNet-CNN-baseline     | 1 | ~65min    | 
VGG16-CNN-baseline     | 2 | ~130min    |
Res50-CNN-baseline     | 1 | ~100min    |
VGG16-HardnessPredictor     | 4 | ~220min   |
Res50-HardnessPredictor     | 2 | ~170min    |

### inference

This part has been presented in Table 1 of the paper.



## References

[1] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra.  Grad-cam:  Visual explanations from deep networks via gradient-based localization.  In Proceedings of the IEEE International Conference on Computer Vision, pages 618–626, 2017.

[2] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pages 3319–3328. JMLR. org,4662017.

[3] Pei Wang and Nuno Vasconcelos. Towards realistic predictors. In The European Conference on Computer Vision, 2018.

[4] Welinder P., Branson S., Mita T., Wah C., Schroff F., Belongie S., Perona, P. “Caltech-UCSD Birds 200”. California Institute of Technology. CNS-TR-2010-001. 2010.

[5] Bolei  Zhou,  Hang  Zhao,  Xavier  Puig,  Sanja  Fidler,  Adela  Barriuso,  and  Antonio  Torralba.   Scene parsing through ade20k
