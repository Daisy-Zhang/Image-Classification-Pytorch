# Pytorch Image Classification
General image classification framework implemented by Pytorch for almost all image classification task.

## structure

### /checkpoints

This directory will store all models you trained.

### /data

Please put all your training or test data in this directory and follow the original directory structure. I provide two kinds of dataset format: Custom and ImageFolder. Users can choose one of them for your convinence. For example, if you want to use ImageFolder, you can put all your training images in **/data/ImageFolder/train** and your test images in **/data/ImageFolder/test**. The further child directory in **/data/ImageFolder/train** is the classes folder of your images, such as **/data/ImageFolder/train/ClassA**...

### /log

All log record file will be stored in this directory.

### /models

You can put all the network model you design in this directory. I already provide three classic networks: **VGG16**, **ResNet18**,**ResNet50**, **GoogLeNet**.

## env

You could use following command to install all dependencies:

```python
pip -r requirements.txt
```

PS: for the **pytorch** version, early version may still be available.

## data

I provide two kinds of dataloader in **dataset.py**: **ImageFolder dataloader** and **Custom dataloader**. Users can choose any one of them. All you need is change following part in **train.py** and **test.py**:

```python
train_loader = get_imagefolder_train_loader()
#train_loader = get_custom_train_loader()
print('get train loader done')
val_loader = get_imagefoler_val_loader()
#val_loader = get_custom_val_loader()
print('get val loader done')
```

And do not forget put your images in /data directory following original structure I provide.

## config

Users can change the config setting in **conf.py** as they need, such as IMAGE_SIZE, EPOCH et. al.

## train

Users can use following command to start training model:

```python
python train.py -model=vgg16 -gpu
```

* **-model**: choose one model in /models.
* **-gpu**: use gpu to train model.

PS: Users can also use following command to get usage:

```python
python train.py -h
```

## test

Users can use following command to evaluate model:

```python
python -model=vgg16 -weights=YOUR_WEIGHT_PATH -gpu -data_path=YOUR_TEST_DATA_PATH
```

* **-model**: choose the model structure of your trained model.
* **-weights**: the model weights path.
* **-gpu**: use gpu to test model.
* **-data_path**: the test data path. Users can modify the **my_eval()** function in **utils.py** to better test your own test data.

PS: Users can also use following command to get usage:

```python
python test.py -h
```

## others

**utils.py**: some utils function used in train.py and test.py. Users can modify this file for their convinence.

If this repo do you a favor, a star is my pleasure :)

And if you find any problem, please contact me or open an issue.
