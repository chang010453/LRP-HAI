#!/bin/bash
#./experiments/scripts/test_LRP_HAI.sh 0 pascal_voc_0712 1 0 0
set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3
NBR_FIX=$4 # <= 0: auto-stop; >= 1: enforce exactly that nbr fixations / image
ALPHA=$5	# use attention's alpha as fixation location

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:5:$len}
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
    P4=True
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
    P4=True
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
    P4=True
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
    P4=True
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
    P4=True
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

# Set up base weights paths according to your own system
WEIGHTS_PATH=/home/user/LRP-HAI/LRP-HAI-training-results/experiment/original/${DATASET}/${NET}/original-1/output/${NET}_LRP_HAI/voc_2007_trainval+voc_2012_trainval/${NET}_LRP_HAI_iter_110000.ckpt


if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net_drl.py \
    --imdb ${TEST_IMDB} \
    --model ${WEIGHTS_PATH} \
    --cfg experiments/cfgs/LRP-HAI-${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --nbr_fix ${NBR_FIX} \
	  --alpha ${ALPHA} \
    --set NBR_CLASSES ${NBR_CLASSES} P4 ${P4}\
       ANCHOR_SIZES ${ANCHOR_SIZES} ANCHOR_STRIDES ${ANCHOR_STRIDES} ANCHOR_RATIOS ${ANCHOR_RATIOS} \
       ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net_drl.py \
    --imdb ${TEST_IMDB} \
    --model ${WEIGHTS_PATH} \
    --cfg experiments/cfgs/LRP-HAI-${NET}.yml \
    --net ${NET} \
    --nbr_fix ${NBR_FIX} \
	  --alpha ${ALPHA} \
    --set NBR_CLASSES ${NBR_CLASSES} P4 ${P4}\
       ANCHOR_SIZES ${ANCHOR_SIZES} ANCHOR_STRIDES ${ANCHOR_STRIDES} ANCHOR_RATIOS ${ANCHOR_RATIOS} \
       ${EXTRA_ARGS}
fi
