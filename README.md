# Image classification based on Convolutional Autoencoder and CNN 
A PyTorch implementation of Convolutional autoencoder (CAE) and CNN on cifar-10.

The default configuration of this repository jointly trains CAE and CNN at the same time. The training scheme is presented below.

<img src="https://github.com/Ryu0w0/meta_repository/blob/master/ConvAutoencoder/images/Structure_CAE_CNN.PNG" width=60%>

## How to run
1. Call autoencoder_main.py
2. Log files and tensorboard files recording acc and loss are produced under the directory of ./files/output

## Configuration
There are 2 locations to modify to change the behavior of the model and training.

- Json files under `./files/input/models/configs/*.json`
    - "use_cnn": true if training CNN
    - "use_cae": true if training CAE
    - "only_train_cae_until": Only CAE is trained while specified number of epoch here (int). Afterward, CNN is jointed with CAE and the two models are trained. 
    - "train_data_regulation": This parameter is map such that `{cls_nm: data_num, ...}`. The number of training data per class is regulated according to this parameter.
    - "train_data_expansion": This parameter is map such that `{cls_nm: data_num, ...}`. The number of training data per class is oversampled according to this parameter.
    - "cnn"
        - "block_sizes": This parameter is a list of lists such that [[1st_conv_depth_from_in(int), to(int), use_max_pooling(boolean)], [2nd,...,...]],
        - "use_mixed_input": true if encoded feature from CAE is inserted into the first feature maps, otherwise, encoded features are used as input of CNN.
    - "cae":
        - "enc_block_sizes": This parameter is a list of lists such that [[1st_conv_depth_from_in(int), to(int)], [2nd,...,...]]
        - "dec_block_sizes": Same with the enc_block_sizes.

- Arguments specified in `autoencoder_main.py`
    - An important argument is `model_config_key`, corresponding to the file name of Json file explained above.
    - Other arguments can be used as default. Specification is provided in `autoencoder_main.py`.
    
## Repository composition of main directories
- dataset: Abstract and sub-class of Cifar-10
- files:
    - input/dataset: Location where Cifar-10 data file is downloaded
    - input/models/configs: There are configuration files controlling model structure and etc
    - output/board: Tensorboard files are produced
    - output/logs: Log files are outputted
    - output/images: Input and output images of CAE are produced
- models: Model definition of CAE and CNN as well as a whole network (classifier.py) of CAE and CNN
- trainer: Abstract class and sub-class of model trainer
