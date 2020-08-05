# PFLD_68Points_Pytorch

Implementation of PFLD For 68 Facial Landmarks By Pytorch

### DataSets
- **WFLW Dataset**  

  [Wider Facial Landmarks in-the-wild (WFLW)](https://wywu.github.io/projects/LAB/WFLW.html) is a new proposed face dataset. It contains 10000 faces (7500 for training and 2500 for testing) with 98 fully manual annotated landmarks.   

  1.Training and Testing images[[Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)][[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)], Unzip and put to `./data/WFLW/raw/`

  2.Have got `list_68pt_rect_attr_train.txt` and `list_68pt_rect_attr_test.txt`. If you want to get them by youself, please watch [get68psFrom98psWFLW.py](https://github.com/github-luffy/PFLD_68points_Pytorch/blob/master/data/WFLW/get68psFrom98psWFLW.py) and run it before please get WFLW [Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz) , unzip and put to `./data/WFLW/`   

  3.move `Mirror68.txt` to `./data/WFLW/annotations/`
  
   ~~~shell
    $ cd ./data/WFLW 
    $ python3 WFLW_SetPreparation68.py
   ~~~
 
- **300W Dataset**

  [300W](https://ibug.doc.ic.ac.uk/resources/300-W/) is a very general face alignment dataset. It has a total of 3148+689 images, each image contains more than one face, but only one face is labeled for each image.File directory includes afw(337)，helen(train 2000+test 330)，ibug(135)，lfpw(train 811+test 224) with 68 fully manual annotated landmarks.

  1.Training and Testing images[[Databases](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)][[Baidu Drive](https://pan.baidu.com/s/1A41fnQPFMFgmUsqEwb4m6A)], Unzip and put to `./data/300W/raw/`

  2.Have got `list_68pt_rect_attr_train.txt` and `list_68pt_rect_attr_test.txt`. If you want to get them by youself, please watch [get68pointsfor300W.py](https://github.com/github-luffy/PFLD_68points_Pytorch/blob/master/data/300W/get68pointsfor300W.py) and run it  

  3.move `Mirror68.txt` to `./data/300W/annotations/`
  
   ~~~shell
    $ cd ./data/300W 
    $ python3 300W_SetPreparation68.py
   ~~~
 
- **300VW Dataset** 

- **Your Own Dataset**  

  If you want to get facial landmarks for new face data, please use [Detect API](https://www.faceplusplus.com.cn/face-detection/#demo) of face++. For specific operations,  
  
  please refer to [API Document](https://console.faceplusplus.com.cn/documents/4888373).And refer to `./data/getNewFacialLandmarksFromFacePP.py` for using  the api interface.  
  
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

  testing:

  ~~~shell
   $ python3 camera.py
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
  
### pytorch -> onnx -> ncnn

**Pytorch -> onnx -> onnx_sim**  

  make sure pip3 install onnx-simplifier

  ~~~~shell
   $ python3 pytorch2onnx.py
   $ python3 -m onnxsim model.onnx model_sim.onnx
  ~~~~

**onnx_sim -> ncnn**  

  how to build :https://github.com/Tencent/ncnn/wiki/how-to-build
  
  ~~~shell
   $ cd ncnn/build/tools/onnx
   $ ./onnx2ncnn model_sim.onnx model_sim.param model_sim.bin
  ~~~

### reference: 

PFLD: A Practical Facial Landmark Detector https://arxiv.org/pdf/1902.10859.pdf

Tensorflow Implementation for 98 Facial Landmarks: https://github.com/guoqiangqi/PFLD
