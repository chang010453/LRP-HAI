#!/bin/bash
#./experiments/scripts/train_LRP_HAI.sh 0 pascal_voc_0712 20000 110000 True
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
DET_START=$4 # when to start detector-tuning (alternate policy, detector training, I used 20000)
ITERS=$5 # number of iterations (images to iterate) in training
ALPHA=$6	# use attention's alpha as fixation location

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:6:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    LRP_HAI_STEPSIZE="90000"
    NBR_CLASSES="21"
    ANCHOR_SIZES="[128,256,512]"
    ANCHOR_STRIDES="[16,]"
    ANCHOR_RATIOS="[0.5,1,2]"
    P4=False
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    LRP_HAI_STEPSIZE="90000"
    NBR_CLASSES="21"
    ANCHOR_SIZES="[128,256,512]"
    ANCHOR_STRIDES="[16,]"
    ANCHOR_RATIOS="[0.5,1,2]"
    P4=False
    ;;
  pascal_voc_0712_test)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval+voc_2007_test"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[80000]"
    LRP_HAI_STEPSIZE="90000"
    NBR_CLASSES="21"
    ANCHOR_SIZES="[128,256,512]"
    ANCHOR_STRIDES="[16,]"
    ANCHOR_RATIOS="[0.5,1,2]"
    P4=False
    ;;
  cell)
    TRAIN_IMDB="cell_train"
    TEST_IMDB="cell_val"
    STEPSIZE="[80000]"
    LRP_HAI_STEPSIZE="90000"
    NBR_CLASSES="8"
    ANCHOR_SIZES="[64,128,256,512]"
    ANCHOR_STRIDES="[16,]"
    ANCHOR_RATIOS="[0.5,1,2]"
    P4=False
    ;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[400000]"
    LRP_HAI_STEPSIZE="390000"
    NBR_CLASSES="81"
    ANCHOR_SIZES="[64,128,256,512]"
    ANCHOR_STRIDES="[16,]"
    ANCHOR_RATIOS="[0.5,1,2]"
    P4=False
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

# Set up paths according to your own system
# Below SAVE_PATH is used when saving trained weights, whereas WEIGHTS_PATH
# is used for loading existing weights
SAVE_PATH=/home/user/LRP-HAI/LRP-HAI-training-results/experiment/drl-model-2/${DATASET}/${NET}/drl-model-2-1/
WEIGHTS_PATH=/home/user/LRP-HAI/LRP-HAI-training-results/fr-rcnn-weights/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_180000.ckpt

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net_drl.py \
    --weight ${WEIGHTS_PATH} \
    --save ${SAVE_PATH} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${ITERS} \
    --cfg experiments/cfgs/LRP-HAI-${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --det_start ${DET_START} \
    --alpha ${ALPHA} \
    --set ANCHOR_SIZES ${ANCHOR_SIZES} ANCHOR_STRIDES ${ANCHOR_STRIDES} ANCHOR_RATIOS ${ANCHOR_RATIOS} \
          NBR_CLASSES ${NBR_CLASSES} TRAIN.STEPSIZE ${STEPSIZE} P4 ${P4}\
          LRP_HAI_TRAIN.STEPSIZE ${LRP_HAI_STEPSIZE} ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net_drl.py \
    --weight ${WEIGHTS_PATH} \
    --save ${SAVE_PATH} \
    --imdb ${TRAIN_IMDB} \
    --imdbval ${TEST_IMDB} \
    --iters ${ITERS} \
    --cfg experiments/cfgs/LRP-HAI-${NET}.yml \
    --net ${NET} \
    --det_start ${DET_START} \
    --alpha ${ALPHA} \
    --set ANCHOR_SIZES ${ANCHOR_SIZES} ANCHOR_STRIDES ${ANCHOR_STRIDES} ANCHOR_RATIOS ${ANCHOR_RATIOS} \
          NBR_CLASSES ${NBR_CLASSES} TRAIN.STEPSIZE ${STEPSIZE} P4 ${P4}\
          LRP_HAI_TRAIN.STEPSIZE ${LRP_HAI_STEPSIZE} ${EXTRA_ARGS}
fi
