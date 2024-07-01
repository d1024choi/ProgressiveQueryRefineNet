# The official implementation of "Progressive Query Refinement Framework for Bird's-Eye-View Semantic Segmentation from Surrounding Images" to be presented in IROS 2024.
![](GIF/HLS.gif)

## Note

+ ...

## Setup

+ **Config Setting** : edit ./config/config.json such that "dataset_dir" correctly locates the directory where your nuScenes dataset is stored. In addition, add your targets (e.g., 'vehicle', 'road', 'lane', 'pedestrian') into "target" in ./config/Scratch/data.json. For example, if you set "target":["vehicle", "pedestrian"], the model will automatically be configured and trained to predict both 'vehicle' and 'pedestrian'.


+ **Implemenation Environment** : The model is implemented by using Pytorch. We share our anaconda environment in the folder 'anaconda_env'. We trained our model on a server equipped with 4 NVIDIA GeForce RTX 4090 graphic cards. To run the deformable attention of BEVFormer, you need to install CUDA first (the version 11.1 is installed in our server) and then compile dedicated CUDA operators in ./models/ops as follows.
```sh
$ cd ./models/ops
$ sh ./make.sh
$ python test.py # unit test (should see all checking is True)
```

## Train and Test New Models
To train the model from scratch, run the followings. The network parameters of the trained models will be stored in the folder ***saved_models***.
```sh
$ sh run_train.sh
```

**argumentparser.py** have a number of command-line flags that you can use to configure the model architecture, hyperparameters, and input / output settings. You can find the descriptions in the file.


To test the trained model, first edit the parameter 'exp_id' in 'run_test.sh' file to match your experiment id and run the followings. You also need to set 'target' in the file to one of four categories (vehicle, pedestrian, road, lane)
```sh
$ sh run_test.sh
```

## Test Pre-trained Models
To test the pre-trained models, first download the pre-trained model parameters from [Here](https://drive.google.com/drive/folders/1YgG7bUVHvc0-WWnLt1vL3HIIjjW2lHFt?usp=sharing). Next, copy them into 'saved_models' folder. Finally, edit the parameter 'exp_id' in 'run_test.sh' file to match the downloaded experiment id and run the followings. It is worth nothing that we trained our models four times each of which corresponds to one of the four classes.
```sh
$ python run_test.sh
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
