#!/bin/bash

settings=settings.py
model_loader=loader/model_loader.py

cp $settings $settings'.bkp'
sed -i '5c MODEL = "vgg16"' $settings
sed -i '6c DATASET = "imagenet"' $settings
sed -i '87c \ \ \ \ BATCH_SIZE = 1' $settings
sed -i "94c PRE_PROCESS = 'caffe'" $settings

cp $model_loader $model_loader'.bkp'
cp experiments/test_faster_rcnn_vgg16.py $model_loader
sed -i '13c OUTPUT_FOLDER = "result/pytorch_"+MODEL+"_pascal_det"' $settings
sed -i "65c \ \ \ \ \ \ \ \ MODEL_FILE = '../my-faster-rcnn/results/result-31-n/param-16-80176.pth'" $settings
python main.py
rm $model_loader
mv $model_loader'.bkp' $model_loader

sed -i '13c OUTPUT_FOLDER = "result/pytorch_"+MODEL+"_imagenet_cls"' $settings
sed -i "65c \ \ \ \ \ \ \ \ MODEL_FILE = '../my-faster-rcnn/data/vgg16_caffe.pth'" $settings
python main.py

rm $settings
mv $settings'.bkp' $settings