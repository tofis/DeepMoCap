# DeepMoCap: Optical Motion Capture leveraging multiple Depth Sensors, Retro-reflectors and Fully Convolutional Neural Networks
By [Anargyros Chatzitofis](https://www.iti.gr/iti/people/Anargyros_Chatzitofis.html), [Dimitris Zarpalas](https://www.iti.gr/iti/people/Dimitrios_Zarpalas.html), [Stefanos Kollias](https://www.ece.ntua.gr/gr/staff/15), [Petros Daras](https://www.iti.gr/iti/people/Petros_Daras.html).

## Introduction
In this paper, a low-cost, marker-based optical motion capture method is proposed, namely DeepMoCap, using multiple spatio-temporally aligned infrared-depth sensors and retro-reflective straps and patches (reflectors). 

DeepMoCap explores motion capture by automatically localizing and labeling reflectors on depth images and, subsequently, on 3D space. Introducing a non-parametric representation to encode the temporal correlation among pairs of colorized depth and 2.5D optical flow frames, a multi-stage fully Convolutional Neural Network (FCNN) architecture is proposed to jointly learn reflector locations and their temporal dependency among sequential frames. The extracted reflector 2D locations are spatially mapped in 3D space, resulting in robust optical data extraction. To this end, the subject's motion is efficiently captured by applying a template-based fitting technique. 

Two datasets have been created and made publicly available for evaluation purposes; one comprising multi-view depth and 2.5D optical flow annotated images (DMC2.5D) and a second, consisting of spatio-temporally aligned multi-view depth images along with inertial and ground truth MoCap data (DMC3D). The FCNN model outperforms its competitors on the DMC2.5D dataset, while the motion capture outcome is evaluated against RGB-D and inertial fusion approaches on DMC3D, outperforming the next best method by $4.5\%$ in total 3D PCK accuracy.

<!-- <p align="left">
<img src="https://github.com/ZheC/Multi-Person-Pose-Estimation/blob/master/readme/dance.gif", width="720">
</p>

<p align="left">
<img src="https://github.com/ZheC/Multi-Person-Pose-Estimation/blob/master/readme/shake.gif", width="720">
</p> -->

This project is licensed under the terms of the [license](LICENSE).



## Contents
1. [Testing](#testing)
2. [Citation](#citation)

## Testing
<!-- 
### C++ (realtime version, for demo purpose)
- Please use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), now it can run in CPU/ GPU and windows /Ubuntu.
- Three input options: images, video, webcam

### Matlab (slower, for COCO evaluation)
- Compatible with general [Caffe](http://caffe.berkeleyvision.org/). Compile matcaffe. 
- Run `cd testing; get_model.sh` to retrieve our latest MSCOCO model from our web server.
- Change the caffepath in the `config.m` and run `demo.m` for an example usage.

### Python
- `cd testing/python`
- `ipython notebook`
- Open `demo.ipynb` and execute the code

## Training

### Network Architecture
![Teaser?](https://github.com/ZheC/Multi-Person-Pose-Estimation/blob/master/readme/arch.png)

### Training Steps 
- Run `cd training; bash getData.sh` to obtain the COCO images in `dataset/COCO/images/`, keypoints annotations in `dataset/COCO/annotations/` and [COCO official toolbox](https://github.com/pdollar/coco) in `dataset/COCO/coco/`. 
- Run `getANNO.m` in matlab to convert the annotation format from json to mat in `dataset/COCO/mat/`.
- Run `genCOCOMask.m` in matlab to obatin the mask images for unlabeled person. You can use 'parfor' in matlab to speed up the code.
- Run `genJSON('COCO')` to generate a json file in `dataset/COCO/json/` folder. The json files contain raw informations needed for training.
- Run `python genLMDB.py` to generate your LMDB. (You can also download our LMDB for the COCO dataset (189GB file) by: `bash get_lmdb.sh`)
- Download our modified caffe: [caffe_train](https://github.com/CMU-Perceptual-Computing-Lab/caffe_train). Compile pycaffe. It will be merged with caffe_rtpose (for testing) soon.
- Run `python setLayers.py --exp 1` to generate the prototxt and shell file for training.
- Download [VGG-19 model](https://gist.github.com/ksimonyan/3785162f95cd2d5fee77), we use it to initialize the first 10 layers for training.
- Run `bash train_pose.sh 0,1` (generated by setLayers.py) to start the training with two gpus.  -->

## Citation
Please cite the paper in your publications if it helps your research:

    
    
    @inproceedings{cao2017realtime,
      author = {Zhe Cao and Tomas Simon and Shih-En Wei and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields},
      year = {2017}
      }
	  
    @inproceedings{wei2016cpm,
      author = {Shih-En Wei and Varun Ramakrishna and Takeo Kanade and Yaser Sheikh},
      booktitle = {CVPR},
      title = {Convolutional pose machines},
      year = {2016}
      }