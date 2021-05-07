# Pytorch_Autoencoder_Example
Autoencoder which acts same as traditional 3 bit encoder

## 0. Experiment Motivation

Following is a network representation.

![autoencoder_diagram](https://user-images.githubusercontent.com/77431192/117440740-6c694100-af6f-11eb-8169-47354897c160.png)
## 1. Prepare Data
Data are already saved as `data.pickle`. The file comprises 8 kinds of one-hot encoded 8 bit data, number of data is 1000 for each kind. 
(details are below)


[1,0,0,0,0,0,0,0] x 1000,
[0,1,0,0,0,0,0,0] x 1000,
.
.
.
[0,0,0,0,0,0,0,1] x 1000

The file is generated by `generate_one_hot_data.py`.

## 2. Train
~~~
python train.py
~~~
pretrained model are kept in `model_ckpt/Autoencoder` directory. 

## 3. Test
~~~
python test.py
~~~

Test results are as follows.
#### 3.1. Raw results

#### 3.2. Processesd results
