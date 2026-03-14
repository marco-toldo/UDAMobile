
### Unsupervised Domain Adaptation for Mobile Semantic Segmentation based on Cycle Consistency and Feature Alignment ###

This is the code of the paper: Toldo M., Michieli U., Agresti G., Zanuttigh P., 'Unsupervised Domain Adaptation for Mobile Semantic Segmentation based on Cycle Consistency and Feature Alignment'

The PDF can be found at: 
The webpage of the paper is:


### Setup ###

1) Download the pre-trained model on the Pascal VOC 2012 of the MobileNetV2 network from: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
2) Download source and target datasets (e.g. GTA and Cityscapes) and build the tfrecords files with the utils/build_data.py script

### Training ###

1) A preliminary training stage of the segmentation network is required. The training on the GTA dataset can be run with the following command:

 python deeplab/train.py \
    --train_tfrecords={PATH_TO_TRAIN_TFRECORDS} \
    --validation_tfrecords={PATH_TO_VAL_TFRECORDS} \
    --pretrained_model_seg={PATH_TO_PRETRAINED_SEG_WEIGHTS}

2) Then the domain adaptation framework can be trained in the GTA-Cityscapes scenario by running:
 
 python deeplab/train.py \
    --X_tfrecords={PATH_TO_SOURCE_TFRECORDS} \
    --Y_tfrecords={PATH_TO_TARGET_TFRECORDS} \
    --validation_tfrecords={PATH_TO_VAL_TFRECORDS} \
    --pretrained_model_seg={PATH_TO_SOURCE_PRETRAINED_SEG_WEIGHTS}
 
 where the new weights from the source pre-trained segmentation network should be loaded. 

 


