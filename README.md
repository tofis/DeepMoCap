# DeepMoCap: Deep Optical Motion Capture using multiple Depth Sensors and Retro-reflectors
By [Anargyros Chatzitofis](https://www.iti.gr/iti/people/Anargyros_Chatzitofis.html), [Dimitris Zarpalas](https://www.iti.gr/iti/people/Dimitrios_Zarpalas.html), [Stefanos Kollias](https://www.ece.ntua.gr/gr/staff/15), [Petros Daras](https://www.iti.gr/iti/people/Petros_Daras.html).



## Introduction
**DeepMoCap** constitutes a low-cost, marker-based optical motion capture method that consumes multiple spatio-temporally aligned infrared-depth sensor streams using retro-reflective straps and patches (reflectors). 

DeepMoCap explores motion capture by automatically localizing and labeling reflectors on depth images and, subsequently, on 3D space. Introducing a non-parametric representation to encode the temporal correlation among pairs of colorized depthmaps and 3D optical flow frames, a multi-stage Fully Convolutional Network (FCN) architecture is proposed to jointly learn reflector locations and their temporal dependency among sequential frames. The extracted reflector 2D locations are spatially mapped in 3D space, resulting in robust optical data extraction. To this end, the subject's motion is efficiently captured by applying a template-based fitting technique.

 ![Teaser?](http://www.deepmocap.com/img/overview.png)
 
 ![Teaser?](http://www.deepmocap.com/img/overall.png)

This project is licensed under the terms of the [license](LICENSE).



## Contents
1. [Testing](#testing)
2. [Datasets](#datasets)
3. [Citation](#citation)

## Testing
For testing the FCN model, please visit ["testing/"](/testing/) enabling the 3D optical data extraction from colorized depth and 3D optical flow input. The data should be appropriately formed and the DeepMoCap FCN model should be placed to ["testing/model/keras"](/testing/model/keras).

The proposed FCN is evaluated on the DMC2.5D dataset measuring mean Average Precision (mAP) for the entire set, based on Percentage of Correct Keypoints (PCK) thresholds (a = 0.05). The proposed method outperforms the competitive methods as shown in the table below.

| Method  | Total | Total (without end-reflectors) |
| :---: | :---: | :---: |
| CPM  | 92.16%  | 95.27% |
| CPM+PAFs  | 92.79\%  | 95.61% |
| CPM+PAFs + 3D OF  | 92.84\%  | 95.67% |
| **Proposed**  | **93.73%**  | **96.77%** |

![Logo](http://www.deepmocap.com/img/3D_all.png)



## Supplementaty material (video)
[![Teaser?](http://www.deepmocap.com/img/video_splash.png)](https://www.youtube.com/watch?v=OvCJ-WWyLcM)

## Datasets
Two datasets have been created and made publicly available for evaluation purposes; one comprising multi-view depth and 3D optical flow annotated images (DMC2.5D), and a second, consisting of spatio-temporally aligned multi-view depth images along with skeleton, inertial and ground truth MoCap data (DMC3D).

### DMC2.5D
The DMC2.5D Dataset was captured in order to train and test the DeepMoCap FCN. It comprises pairs per view of: 
 - colorized depth and 
 - 3D optical flow data (the primal-dual algorithm used in the present work can be found @ https://github.com/MarianoJT88/PD-Flow, using IR data instead of RGB).
 
The samples were randomly selected from 8 subjects. More specifically, 25K single-view pair samples were annotated with over 300K total keypoints (i.e., reflector 2D locations of current and previous frames on the image), trying to cover a variety of poses and movements in the scene. 20K, 3K and 2K samples were used for training, validation and testing the FCN model, respectively. The annotation was semi-automatically realized by applying image processing and 3D vision techniques, while the dataset was manually refined using the [2D-reflectorset-annotator](/tools/2D-reflector-annotator/).

 ![Teaser?](http://www.deepmocap.com/img/DMC2.5D_github.png)

To get the DMC2.5D dataset, please contact the owner of the repository via github or email (tofis@iti.gr).

### DMC3D

![Teaser?](http://www.deepmocap.com/img/depth.png)

The DMC3D dataset consists of multi-view depth and skeleton data as well as inertial and ground truth motion capture data. Specifically, 3 Kinect for Xbox One sensors were used to capture the IR-D and Kinect skeleton data along with 9 **XSens MT inertial** measurement units (IMU) to enable the comparison between the proposed method and inertial MoCap approaches. Further, a **PhaseSpace Impulse X2** solution was used to capture ground truth MoCap data. The preparation of the DMC3D dataset required the spatio-temporal alignment of the modalities (Kinect, PhaseSpace, XSens MTs). The setup used for the Kinect recordings provides spatio-temporally aligned IR-D and skeleton frames.

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

The annotation tool for the spatio-temporally alignment of the 3D data will be publicly available soon.

To get the DMC3D dataset, please contact the owner of the repository via github or email (tofis@iti.gr).

## Citation
This paper has been published in MDPI Sensors, Depth Sensors and 3D Vision Special Issue [[PDF]](https://www.mdpi.com/1424-8220/19/2/282)

Please cite the paper in your publications if it helps your research:    

<pre><code>
@article{chatzitofis2019deepmocap,
  title={DeepMoCap: Deep Optical Motion Capture Using Multiple Depth Sensors and Retro-Reflectors},
  author={Chatzitofis, Anargyros and Zarpalas, Dimitrios and Kollias, Stefanos and Daras, Petros},
  journal={Sensors},
  volume={19},
  number={2},
  pages={282},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
</pre></code>	  
