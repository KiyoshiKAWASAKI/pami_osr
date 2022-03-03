# MSDNet-PyTorch

This repository contains the PyTorch implementation of the paper [Multi-Scale Dense Networks for Resource Efficient Image Classification](https://arxiv.org/pdf/1703.09844.pdf)

Citation:

    @inproceedings{huang2018multi,
        title={Multi-scale dense networks for resource efficient image classification},
        author={Huang, Gao and Chen, Danlu and Li, Tianhong and Wu, Felix and van der Maaten, Laurens and Weinberger, Kilian Q},
        journal={ICLR},
        year={2018}
    }

# Dependencies:

## Please open a new branch based on this branch if you need to modify anything or doing another task with this code.

Please create a new environent using anoconda and install all required packages.

`conda create -n <environment-name> --file requirement.txt`
   

# Data

We use Json files for the data loader. Our data preprocessing is rather complex. Please contact Jin Huang if you need our data file or want to completely reproduce our data, and she will offer more detailed instructions.


# Training and validation

## Change the parameters

In either pipeline.py or known_pipeline.py, look at the following parameters at the very beginning of the scripts:

- use_performance_loss
- use_exit_loss
- cross_entropy_weight
- perform_loss_weight
- exit_loss_weight
- random_seed
- use_modified_loss: 2 different setups for psyphy loss, refer MSDNet-PyTorch/utils/pipeline_util.py line 565
- run_test: False for training phase, True for testing phase

Other changes: save_path_base and all paths for data on your own server.


## Train models

To train models with known known and known unknown data, run following command under root directory:

`bash ./job_scripts/train.sh`

To train models with only known known data, run following command under root directory:

`bash ./job_scripts/train_known.sh`

It will save best models during training. All results will be saved to results.csv


# Testing

1. Set run_test to False and test_model_dir in either pipeline.py or known_pipeline.py, depending on your task.

2. Under root directory, run:

`bash ./job_scripts/test_known.sh` for oepn-set models
or 
`bash ./job_scripts/test.sh` for close-set models

It will save features, labels, probabilities.

3. Under MSDNet-PyTorch/utils:

- Activate your conda environment.
- Change save_path_sub and epoch (epoch index for the best model) in either process_known_results.py or process_results.py
- Run:

`python process_known_results.py`

or

`python process_results.py`



