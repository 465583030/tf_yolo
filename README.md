## A YOLOv3-416 implementation on TensorFlow Low-Level API

## Summary:
* The model was made in pure TensorFlow Low-Level API, detection outputs were processed in NumPy and OpenCV2 was used for image manipulations.
* Non TensorFlow concise easy to use interface for loading, predicting on YOLOv3-416 model. (training to be added very soon)
* Also come with a concise easy to use interface for loading, pre-processing and post-processing images.
* Interface for running on Intel NCS (required installation of Open-VINO)
* Integratable to any applications.
* Code was written as modular as possible.


## What comes next:
1. ~~Frozen model generation and handling, for needlessness of weight file loading every launch time.~~
2. ~~Code to run model on Movidius Raspberry Pi.~~
3. Trainer for custom object detection.

## Instructions:

### Dependencies:
- TensorFlow/TensorFlow-GPU
- NumPy
- OpenCV2
- Jupyter Notebook (only pre-installed in Anaconda)

### Installation:
any of these libraries above can be easily installed with the following command: `pip install package_name` except for TensorFlow-GPU as you have to also install Nvidia CUDA + cuDNN which is a very pessimistic hassle. Being committed to avoid this in every way, I use [Anaconda](https://www.anaconda.com/ "Anaconda Homepage") for ease of package and working environment management, which I will use for this instruction.

If you have never try to install Tensorflow-GPU in the prior way, I would recommend you to try it first to have a taste so you could appreciate the simplicity of the latter ;)

1. download and install Anaconda from its Website.
   from this point I will be using my personal default of naming, you may change it however you like
2. Open Anaconda Prompt (or add the path of Anaconda Executable to sytem variables to use within CMD) and create a vitural environment named `tf-gpu`:
      ```
      conda create --name tf-gpu
      ```
3. activate it
      ```
      conda activate tf-gpu
      ```
4. Install TensorFlow-GPU (leave out the "-GPU" if you are installing Tensorflow non GPU version):
      ```
      conda install -c anaconda tensorflow-gpu
      ```
5. Install OpenCV2:
      ```
      conda install -c conda-forge opencv
      ```

### Usage:
* To detect on images, open file ***detect_images.ipynb*** with `jupyter notebook` and specify either the path to the image or the directory of images to the call of the method `imset_frompath(path)`, then run all the code. Output images will be saved to `outputs/` of the current working directory.
![image](https://user-images.githubusercontent.com/18170028/52904485-ab4b3900-3235-11e9-9a79-c23e94c1bf28.png)
* To detect live on camera, run the following command in prompt/conda prompt:
    ```
    python live.py
    ```
### NCS:

#### Dependencies:
- Open-VINO
- picamera (optionally if you intend to run on rapberry pi equipped with Pi Camera module)
#### Installation:
*This instruction is in Windows but the procedures are the same for other OSes please follow up and refer to the installation instruction from Intel.
- Install [Intel Open-VINO](https://software.intel.com/en-us/openvino-toolkit)
- if:

     1. you have set anaconda to system variables

     2. and used my naming for the virtual environment

     3. install open-vino to C:\ drive

  then just run generate_ir.bat in a command prompt and see Usage.

- else:

     1. Activate your virtual environment (skip if you dont use it)

     2. run the file C:\Intel\computer_vision_sdk\bin\setupvars.bat to source open-vino SDK for Python

     3. run the following command to generate the intermediate representations (IR):
         ```
         ...\YOLOw> python ...\Intel\computer_vision_sdk\deployment_tools\model_optimizer\mo_tf.py --input_model .\frozen_yolow.pb --tensorflow_use_custom_operations_config .\yolow_ir_config.json --batch 1 --data_type FP16 --output_dir .\ir\
         ```
 *Please also output the IR to the ir\ directory since the app will read from it by default.
#### Usage
- Start live detection using the NCS: (on PC/PRi with external camera)
    ```
    python live_ncs.py
    ```
- Start live detection using the NCS: (on RPi with Pi camera module)
    ```
    python live_ncs_rpi.py
    ```
  
## References:
[Original YOLO](https://github.com/pjreddie/darknet/wiki/YOLO:-Real-Time-Object-Detection)

[DarkFlow](https://github.com/thtrieu/darkflow)

[TensorFlow-SLIM YOLOv3](https://github.com/mystic123/tensorflow-yolo-v3)

[Open-VINO](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer)
