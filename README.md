# MSDNet-PyTorch

This is the code for paper: [Measuring Human Perception to Improve Open Set Recognition](https://arxiv.org/abs/2209.03519). 

Please cite this paper if you find it useful. Citation:

    @misc{huang2023measuring,
          title={Measuring Human Perception to Improve Open Set Recognition}, 
          author={Jin Huang and Derek Prijatelj and Justin Dulay and Walter Scheirer},
          year={2023},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI)}

# Dependencies:

For the most updated method, please look at ```debug_loss``` branch.

For training our loss with ResNet, please check out ```resnet``` branch.

Please create a new environent using anoconda and install all required packages.

    `conda create -n <environment-name> --file requirement.txt`
   

# Data

We use Json files for the data loader. All our training and testing data are available on google drive here: https://drive.google.com/drive/folders/1f6unxlmDCHXvxtna5DIeQnIB_H_EHbBf?usp=sharing

If you are interested in how the data was prepared, check out the detail code scripts under ```utils```.


# Training and validation

## Change the parameters

Main changes before training: ```save_path_base``` and all paths for data on your own server.

In ```pipeline_openset.py```, look at the following parameters at the very beginning of the scripts and change them as preferred:

- use_performance_loss
- use_exit_loss
- cross_entropy_weight
- perform_loss_weight
- exit_loss_weight
- random_seed
- use_modified_loss: 2 different setups for psyphy loss, refer utils/pipeline_util.py
- run_test: False for training phase, True for testing phase


## Train Open Set models

To train models that allow open set recognition, run following command under root directory:

    `bash ./job_scripts/train_openset.sh`

It will save best models during training. And all training and validation accuracy will be saved to ```results.csv```


# Testing

1. Set ```run_test``` to ```False``` and ```test_model_dir``` in ```pipeline_openset.py```, depending on your task.

2. Under root directory, run:

    `bash ./job_scripts/test_openset.sh`

    It will save features, labels, probabilities.

3. Under utils folder:

- Activate your conda environment.
- Change save_path_sub and epoch (epoch index for the best model) in process_results.py
- Run the following file in **```resnet```** branch for the updated testing process: `python process_openset_results.py`
