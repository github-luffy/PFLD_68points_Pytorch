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

  1.Training and Testing images[[Databases](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)][[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)], Unzip and put to `./data/300W/raw/`

  2.Have got `list_68pt_rect_attr_train.txt` and `list_68pt_rect_attr_test.txt`. If you want to get them by youself, please watch [get68pointsfor300W.py](https://github.com/github-luffy/PFLD_68points_Pytorch/blob/master/data/300W/get68pointsfor300W.py) and run it  

  3.move `Mirror68.txt` to `./data/300W/annotations/`
  ~~~shell
      $ cd ./data/300W 
      $ python3 300W_SetPreparation68.py
  ~~~
 
- **300VW Dataset** 

### training & testing

training :

~~~shell
$ sh train.sh
~~~

testing:

~~~shell
$ python3 camera.py
~~~

### pytorch -> onnx -> ncnn

**Pytorch -> onnx -> onnx_sim(pip3 install onnx-simplifier)**

~~~~shell
python3 pytorch2onnx.py
python3 -m onnxsim model.onnx model_sim.onnx
~~~~

**onnx_sim -> ncnn**

~~~shell
cd ncnn/build/tools/onnx
./onnx2ncnn model_sim.onnx model_sim.param model_sim.bin
~~~
