# VWW for vehicles

In this project we have developed an edge-AI application for recognizing vehicle (visualwakewords vehicles) in grayscale images. The specific target of this application is the GAP-board family from Greewaves-technologies. 
A trained .tflite network for the task is already in the GAP_PORTING_xBIT/nntool/ folder. To convert the network to GAP code and run it on the platform simulator (gvsoc) is sufficient:
```
make clean all run platform=gvsoc QUANT_BITS=8/16
```
In the following are presented in details the training of the network on a custom visualwakewords task (or even another image classification task) and the steps for GAP platform porting inside the Makefile magic.

## Requirements
> Tensorflow 1.13
>
> GapSDK 3.2 or more

## Table of contents:
  - [Dataset Preparing and training](#dataset-preparing-and-training)
  - [NN on platform with GAPFlow](#nn-on-platform-with-gapflow)
  - [Validation with Platform Emulator](#validation-with-platform-emulator)
  
## Dataset Preparing and training

To generate the training and validation dataset and to train the Neural Network model we used the open source [Image Classification](https://github.com/tensorflow/models/tree/master/research/slim) template released by the tensorflow team in its TF1.x version with the use of slim.
The original code can train a Neural Network model able to signal the presence of persons in the images. After few, very simple code changes we have extended this feature to each [object in the COCO dataset](https://github.com/amikelive/coco-labels/blob/master/coco-labels-2014_2017.txt). With this tutorial, we will guide you through the usage of the modified framework to reproduce our results for vehicles signaling.

### Download COCO and distill visualwakewords

Visualwakewords dataset derives from a distilling process applied to COCO. For each image in the source dataset, the present/not-present label is saved in a separated annotation file wether in the original COCO-annotation for that image is present or not the class of interest, in our case it consist in a list of object ('bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat').
The TF framework provides a script to automatically download (if not already done) and convert the COCO dataset into the visualwakewords on desired class or classes. It produces a dataset in the TF-Record format:
```
python3 slim/download_and_convert.py --dataset_name=visualwakewords 
                                     --dataset_dir=visualwakewords_vehicle
                                     --foreground_class_of_interest='bicycle, car, motorcycle, ...'
                                     --small_object_area_threshold=0.05
                                     --download
                                     --coco_dir=coco
- dataset_name: name of the dataset to download (one of "flowers", "cifar10", "mnist", "visualwakewords")
- dataset_dir: where to store the dataset
in case of "visualwakewords":
- foreground_class_of_interest: list of COCO object which you want to detect in images
- small_object_area_threshold: minimum percentage of area of the interested object to promote the image to the visualwakewords label of true
- download: whether to download the entire coco dataset or not if already downloaded
- coco_dir: if download=False where to store the coco dataset, if download=True where it is stored
```
### Model Training

Now that the dataset is ready we can train a Neural Network on it. For these experiments we used a mobilenet_v1 (with 224x224 input dimensions and width multiplier of 1), the slim/nets folder contains several example model which can be used as well.
```
python3 train_image_classifier.py \
	      --train_dir='vww_vehicle_train_grayscale' \
	      --dataset_name='visualwakewords' \
	      --dataset_split_name=train \
	      --dataset_dir='./visualwakewords_vehicle/' \
	      --log_every_n_steps=100 \
	      --model_name='mobilenet_v1' \
	      --checkpoint_path='./vww_vehicle_train_grayscale/' \
	      --max_number_of_steps=100000   \  
	      --num_clones=2   \ 
	      --use_grayscale
- train_dir: where to store checkpoints and training info
- dataset_name: again the name of the dataset to train with
- dataset_split: which dataset partition to use for training
- dataset_dir: where to find the datasets TF-Records
- model_name: name of the netowork architecture to train (slim/nets/nets_factory,py for a complete list of supported networks)
- checkpoint_path: where to find the checkpoint files to start from, if not given the network will be trained from scratch
- max_number_of_steps: number of training steps
- num_clones: number of GPU to use for training
```

### Floating Point Model evaluation

The model can now be evaluated in its floating point version on the validation dataset:
```
python3 eval_image_classifier.py   \   
	      --checkpoint_path='vww_train_vehicle_grayscale/'   \   
	      --eval_dir='vww_eval_vehicle_grayscale/' \  
      	--dataset_split_name=val   \   
	      --dataset_dir='visualwakewords_vehicle/'  \  
      	--dataset_name='visualwakewords' \  
	      --model_name='mobilenet_v1' \ 	  
	      --use_grayscale
```
The script will evaluate the network accuracy as number of well predicted images divided by the total number of images, the number of false positives and false negatives. All these metrics can be inspected through tensorboard:
```
tensorboard --logdir='vww_eval_vehicle_grayscale'
```

### Model export and freeze

To export the inference graph, i.e. the tensorflow graphdef file for the inference:
```
python3 slim/export_inference_graph.py
  --model_name=mobilenet_v1 \
  --image_size=224 \
  --output_file=./mobilenet_v1_224_grayscale.pb  \
  --use_grayscale
```

If you then want to use the resulting model with your own or pretrained checkpoints as part of a mobile model, you can run the tensorflow built in command freeze_graph to get a graph def with the variables inlined as constants using:
```
freeze_graph \
  --input_graph=./mobilenet_v1_224_grayscale.pb \
  --output_graph=./frozen_mbv1_224_grayscale.pb \
  --input_checkpoint=./vww_train_vehicle_grayscale/model.ckpt-100000 \
  --input_binary=true \
  --output_node_names=MobilenetV1/Predictions/Reshape_1
```


To inspect the graph and get the output_node_names you can use [Netron](https://lutzroeder.github.io/netron/)

## NN on platform with GAPFlow

Now we are ready for the effective deployment on GAP platform. To address this, we will show you the usage of the GAPFlow, a toolchain developed by Greenwaves-technologies for Neural Network porting on their devices.
First of all, we will use the __nntool__ for translating the high level model description into an internal description which will be used by the __Autotiler__. This tool will leverage on the predictable memory access pattern of convolutional neural network to optimize the C-code which runs on the platform.
To do this we need a floating-point tflite model description:
```
tflite_converter --graph_def=./frozen_mbv1_224_grayscale.pb  \
                 --output_file=mbv1_grayscale.tflite  \
                 --input_arrays=input  \
                 --output_arrays=MobilenetV1/Predictions/Reshape_1
```

### nntool
After following the guide in the [GAPSdk](https://github.com/GreenWaves-Technologies/gap_sdk) repository for installation, we can open the generated tflite with nntool:
```
nntool mbv1_grayscale.tflite
```

## Validation with Platform Emulator
