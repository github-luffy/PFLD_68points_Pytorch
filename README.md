# PFLD_68Points_Pytorch

Implementation of PFLD For 68 Facial Landmarks By Pytorch

### DataSets
- **WFLW Dataset**  

  [Wider Facial Landmarks in-the-wild (WFLW)](https://wywu.github.io/projects/LAB/WFLW.html) is a new proposed face dataset. It contains 10000 faces (7500 for training and 2500 for testing) with 98 fully manual annotated landmarks.   

  1.Training and Testing images[[Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)][[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)], Unzip and put to `./data/WFLW/raw/`

  2.Have got `list_68pt_rect_attr_train.txt` and `list_68pt_rect_attr_test.txt`. If you want to get them by youself, please watch [get68psFrom98psWFLW.py](https://github.com/github-luffy/PFLD_68points_Pytorch/blob/master/data/WFLW/get68psFrom98psWFLW.py) and run it before please get WFLW [Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz) , unzip and put to `./data/WFLW/`   

  3.Move `Mirror68.txt` to `./data/WFLW/annotations/`
  
   ~~~shell
    $ cd ./data/WFLW 
    $ python3 WFLW_SetPreparation68.py
   ~~~
 
- **300W Dataset**

  [300W](https://ibug.doc.ic.ac.uk/resources/300-W/) is a very general face alignment dataset. It has a total of 3148+689 images, each image contains more than one face, but only one face is labeled for each image.File directory includes afw(337)，helen(train 2000+test 330)，ibug(135)，lfpw(train 811+test 224) with 68 fully manual annotated landmarks.

  1.Training and Testing images[[Databases](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)][[Baidu Drive](https://pan.baidu.com/s/1A41fnQPFMFgmUsqEwb4m6A)], Unzip and put to `./data/300W/raw/`

  2.Have got `list_68pt_rect_attr_train.txt` and `list_68pt_rect_attr_test.txt`. If you want to get them by youself, please watch [get68pointsfor300W.py](https://github.com/github-luffy/PFLD_68points_Pytorch/blob/master/data/300W/get68pointsfor300W.py) and run it  

  3.Move `Mirror68.txt` to `./data/300W/annotations/`
  
   ~~~shell
    $ cd ./data/300W 
    $ python3 300W_SetPreparation68.py
   ~~~
 
- **300VW Dataset** 
  
  [300VW](https://ibug.doc.ic.ac.uk/resources/300-VW/) is a video format, which needs to be processed into a single frame picture and corresponds to each key point pts file.

  1.Training and Testing images[[Databases](https://ibug.doc.ic.ac.uk/resources/300-VW/)], Unzip and put to `./data/300VW/raw/`

  2.Run [get68psAndImagesFrom300VW.py](https://github.com/github-luffy/PFLD_68points_Pytorch/blob/master/data/300VW/get68psAndImagesFrom300VW.py) to get `list_68pt_rect_attr_train.txt`

  3.Move `Mirror68.txt` to `./data/300VW/annotations/`
  
   ~~~shell
    $ cd ./data/300VW 
    $ python3 get68psAndImagesFrom300VW.py
    $ python3 300VW_SetPreparation68.py
   ~~~
  
- **Your Own Dataset**  

  If you want to get facial landmarks for new face data, please use [Detect API](https://www.faceplusplus.com.cn/face-detection/#demo) of face++. For specific operations,  
  please refer to [API Document](https://console.faceplusplus.com.cn/documents/4888373). And refer to `./data/getNewFacialLandmarksFromFacePP.py` for using  the api interface.  
  
- **All Dataset**

  After completing the steps of each data set above, you can run the code `merge_files.py` directly .  
  
    ~~~shell
     $ cd ./data
     $ python3 merge_files.py
   ~~~
  
### training & testing

  training :

  ~~~shell
   $ sh train.sh
  ~~~

  reading images from a camera to test:

  ~~~shell
   $ python3 camera.py
  ~~~
  
  reading images from a dir to test:
  
  ~~~shell
   $ python3 test.py
  ~~~

### Result

  Sample IMGs:  

  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/2.jpg)
  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/12.jpg)
  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/14.jpg)
  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/16.jpg)
  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/17.jpg)
  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/20.jpg)
  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/67.jpg)
  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/85.jpg)
  ![Image text](https://github.com/github-luffy/PFLD_68points_Pytorch/tree/master/data/Sample_imgs/86.jpg)  
  
  Details about the models are below: 
  
  tip: please install [resnest](https://github.com/zhanghang1989/ResNeSt) to use ResNest models. 
  
  |    *Name*         |*# Params*| *Mean error*|*Failure rate*|*One iteration time(s)*|
  |:-----------------:|:--------:|:-----------:|:------------:|:---------------------:|
  |    `ResNest50`    |  122.27M |    0.046    |     0.038    |          0.304        |
  | `MobileNetV2_0.25`|   1.09M  |    0.075    |     0.174    |          0.154        |
  | `MobileNetV2_1.00`|   7.28M  |    0.065    |     0.127    |          0.203        |
  |  `BlazeLandmark`  |   7.52M  |    0.069    |     0.131    |          0.171        |
  |     `HRNetV2`     |  545.07M |    0.066    |     0.125    |          0.769        |
  | `efficientnet-b0` |  16.67M  |    0.064    |     0.119    |          0.202        |
  | `efficientnet-b1` |  26.37M  |    0.075    |     0.149    |          0.252        |
  | `efficientnet-b2` |  30.85M  |    0.071    |     0.145    |          0.266        |
  | `efficientnet-b3` |  42.29M  |    0.099    |     0.136    |          0.307        |
  | `efficientnet-b4` |  68.34M  |    0.077    |     0.164    |          0.375        |
  | `efficientnet-b5` |  109.34M |    0.094    |     0.173    |          0.501        |
  | `efficientnet-b6` |  156.34M |    0.081    |     0.175    |          0.702        |
  | `efficientnet-b7` |  244.03M |    0.081    |     0.196    |          0.914        |
  
### pytorch -> onnx -> ncnn

**Pytorch -> onnx -> onnx_sim**  

  Make sure pip3 install onnx-simplifier

  ~~~~shell
   $ python3 pytorch2onnx.py
   $ python3 -m onnxsim model.onnx model_sim.onnx
  ~~~~

**onnx_sim -> ncnn**  

  How to build :https://github.com/Tencent/ncnn/wiki/how-to-build
  
  ~~~shell
   $ cd ncnn/build/tools/onnx
   $ ./onnx2ncnn model_sim.onnx model_sim.param model_sim.bin
  ~~~

### reference: 

PFLD: A Practical Facial Landmark Detector https://arxiv.org/pdf/1902.10859.pdf

ResNest: Split-Attention Networks https://hangzhang.org/files/resnest.pdf 

EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks https://arxiv.org/pdf/1905.11946.pdf

pytorch：https://github.com/lukemelas/EfficientNet-PyTorch

tensorflow：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet

keras: https://github.com/qubvel/efficientnet

Tensorflow Implementation for 98 Facial Landmarks: https://github.com/guoqiangqi/PFLD
