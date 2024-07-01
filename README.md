# The official implementation of "Progressive Query Refinement Framework for Bird's-Eye-View Semantic Segmentation from Surrounding Images" to be presented in IROS 2024.
![](GIF/HLS.gif)

## Note

+ ...

## Setup

+ **Config Setting** : edit ./config/config.json such that "dataset_dir" correctly locates the directory where your nuScenes dataset is stored.


+ **Implemenation Environment** : The model is implemented by using Pytorch. We share our anaconda environment in the folder 'anaconda_env'. We trained our model on a server equipped with 4 NVIDIA GeForce RTX 4090 graphic cards. To run the deformable attention of BEVFormer, you need to install CUDA first (version 11.1 is installed in our server) and then compile dedicated CUDA operators in ./models/ops.
```sh
$ cd ./models/ops
$ sh ./make.sh
$ python test.py # unit test (should see all checking is True)
```

## Train and Test New Models
To train the model from scratch, run the followings. The network parameters of the trained models will be stored in the folder ***saved_models***.
```sh
$ sh nuscenes_train.sh
$ sh argoverse_train.sh
```

**argumentparser.py** have a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings. You can find the descriptions in the file.


To test the trained model, first edit the parameter 'exp_id' in 'nuscenes_test.sh' and 'argoverse_test.sh' files to match your experiment id and run the followings.
```sh
$ sh nuscenes_test.sh
$ sh argoverse_test.sh
```

## Test Pre-trained Models
To test the pre-trained models, first download the pre-trained model parameters from https://drive.google.com/file/d/1kEI3jLueqVejvim_Moh4909yBFQG4jaF/view?usp=sharing. Next, copy them into 'saved_models' folder. Finally, edit the parameter 'exp_id' in 'nuscenes_test.sh' and 'argoverse_test.sh' files to match the downloaded experiment id and run the followings.
```sh
$ python nuscenes_test.sh
$ python argoverse_test.sh
```

## Paper Download
...

## Citation
```
@InProceedings{Choi,
 author = {D. Choi and J. Kang and T. An and K. An and K. Min},
 title = {Progressive Query Refinement Framework for Bird's-Eye-View Semantic Segmentation from Surrounding Images},
 booktitle = {Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
 year = {2024}
}
```
