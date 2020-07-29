# PFLD_68Points_Pytorch

Implementation of PFLD For 68 Facial Landmarks By Pytorch

### DataSets
- **WFLW Dataset Download**  

  [Wider Facial Landmarks in-the-wild (WFLW)](https://wywu.github.io/projects/LAB/WFLW.html) is a new proposed face dataset. It contains 10000 faces (7500 for training and 2500 for testing) with 98 fully manual annotated landmarks.   

  1.Training and Testing images[[Google Drive](https://drive.google.com/file/d/1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC/view?usp=sharing)][[Baidu Drive](https://pan.baidu.com/s/1paoOpusuyafHY154lqXYrA)], Unzip and put to `./data/raw/`  

  2.WFLW [Face Annotations](https://wywu.github.io/projects/LAB/support/WFLW_annotations.tar.gz) , unzip and put to `./data/`  

  3.Run [***./data/get68psFrom98psWFLW.py***](https://github.com/github-luffy/PFLD_68points_Pytorch/blob/master/data/get68psFrom98psWFLW.py), get `list_68pt_rect_attr_train.txt` and `list_68pt_rect_attr_test.txt`.  

  4.move `Mirror68.txt` to `./data/annotations/`  
  ~~~shell
      $ cd ./data 
      $ python3 get68psFrom98psWFLW.py
      $ python3 WFLW_SetPreparation68.py
  ~~~
 
- **300W**

- **300VW** 

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
