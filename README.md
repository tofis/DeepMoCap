# DeepMoCap: Deep Optical Motion Capture using multiple Depth Sensors and Retro-reflectors
By [Anargyros Chatzitofis](https://www.iti.gr/iti/people/Anargyros_Chatzitofis.html), [Dimitris Zarpalas](https://www.iti.gr/iti/people/Dimitrios_Zarpalas.html), [Stefanos Kollias](https://www.ece.ntua.gr/gr/staff/15), [Petros Daras](https://www.iti.gr/iti/people/Petros_Daras.html).



## Introduction
**DeepMoCap** constitutes a low-cost, marker-based optical motion capture method that consumes multiple spatio-temporally aligned infrared-depth sensor streams using retro-reflective straps and patches (reflectors). 

DeepMoCap explores motion capture by automatically localizing and labeling reflectors on depth images and, subsequently, on 3D space. Introducing a non-parametric representation to encode the temporal correlation among pairs of colorized depthmaps and 3D optical flow frames, a multi-stage Fully Convolutional Network (FCN) architecture is proposed to jointly learn reflector locations and their temporal dependency among sequential frames. The extracted reflector 2D locations are spatially mapped in 3D space, resulting in robust optical data extraction. To this end, the subject's motion is efficiently captured by applying a template-based fitting technique.

This project is licensed under the terms of the [license](LICENSE).



## Contents
1. [Testing](#testing)
2. [Datasets](#datasets)
3. [Citation](#citation)

## Testing
![Logo](http://www.deepmocap.com/img/3D_all.png)

## Supplementaty material (video)
[![Teaser?](http://www.deepmocap.com/img/video_splash.png)](https://www.dropbox.com/s/y0iyv2hg5eufl4y/DeepMoCap_vid.mp4?dl=0)

## Datasets
Two datasets have been created and made publicly available for evaluation purposes; one comprising multi-view depth and 3D optical flow annotated images (DMC2.5D), and a second, consisting of spatio-temporally aligned multi-view depth images along with skeleton, inertial and ground truth MoCap data (DMC3D).

### DMC2.5D
The DMC2.5D Dataset was captured in order to train and test the DeepMoCap FCN. It comprises pairs per view of: 
 - colorized depth and 
 - 3D optical flow data.
 
 The samples were randomly selected from 8 subjects. More specifically, 25K single-view pair samples were annotated with over 300K total keypoints (i.e., reflector 2D locations of current and previous frames on the image), trying to cover a variety of poses and movements in the scene. 20K, 3K and 2K samples were used for training, validation and testing the FCN model, respectively. The annotation was semi-automatically realized by applying image processing and 3D vision techniques, while the dataset was manually refined using the [2D-reflectorset-annotator](/tools/2D-reflector-annotator/).

 ![Teaser?](http://www.deepmocap.com/img/DMC2.5D_github.png)

### DMC3D

The DMC3D dataset consists of multi-view IR-D and skeleton data as well as inertial and ground truth motion capture data. Specifically, 3 Kinect for Xbox One sensors were used to capture the IR-D and Kinect skeleton data along with 9 XSens MT \cite{paulichxsens} inertial measurement units (IMU) to enable the comparison between the proposed method and inertial MoCap approaches based on \cite{destelle2014low}. Further, a PhaseSpace Impulse X2 \cite{phasespace} solution was used to capture ground truth MoCap data. The preparation of the DMC3D dataset required the spatio-temporal alignment of the modalities (Kinect, PhaseSpace, XSens MTs). The setup \cite{alexiadis2017integrated} used for the Kinect recordings provides spatio-temporally aligned IR-D and skeleton frames.

|   Exercise    | # of repetitions  | # of frames  |  Type  |
|  :---: |  :---: |  :---: |  :---: |
| Walking on the spot | 10-20 | 200-300 | Free |
| Single arm raise | 10-20 | 300-500 | Bilateral |
| Elbow flexion | 10-20 | 300-500 | Bilateral |
| Knee flexion | 10-20 | 300-500 | Bilateral |
| Closing arms above head | 6-12 | 200-300 | Free |
| Side steps | 6-12 | 300-500 | Bilateral | 
| Jumping jack | 6-12 | 200-300 | Free |
| Butt kicks left-right | 6-12 | 300-500 | Bilateral |
| Forward lunge left-right | 4-10 | 300-500 | Bilateral |
| Classic squat | 6-12 | 200-300 | Free |
| Side step + knee-elbow | 6-12 | 300-500 | Bilateral |
| Side reaches | 6-12 | 300-500 | Bilateral |
| Side jumps | 6-12 | 300-500 | Bilateral |
| Alternate side reaches | 6-12 | 300-500 | Bilateral |
| Kick-box kicking | 2-6 | 200-300 | Free |

## Citation

The paper is currently under review.

<!-- Please cite the paper in your publications if it helps your research:    
    
    @inproceedings{deepmocap2018chatzitofis,
      author = {Anargyros Chatzitofis and Dimitrios Zarpalas and Stefanos Kollias and Petros Daras},
      booktitle = {Sensors},
      title = {DeepMoCap: Optical Motion Capture leveraging multiple Depth Sensors, Retro-reflectors and Fully Convolutional Neural Networks},
      year = {2018}
      } -->
	  
