#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
DATASET=$2
NET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case ${DATASET} in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[50000]"
    ITERS=70000
    ANCHOR_SIZES="[128,256,512]"
    RATIOS="[0.5,1,2]"
    ANCHOR_STRIDES="[16,]"
    P4=True
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    STEPSIZE="[100000]"
    ITERS=180000
    ANCHOR_SIZES="[128,256,512]"
    RATIOS="[0.5,1,2]"
    ANCHOR_STRIDES="[16,]"
    P4=True
    ;;
 cell)
    TRAIN_IMDB="cell_train"
    TEST_IMDB="cell_val"
    STEPSIZE="[100000]"
    ITERS=180000
    ANCHOR_SIZES="[64,128,256,512]"
    RATIOS="[0.5,1,2]"
    ANCHOR_STRIDES="[16,]"
    P4=True
	;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    STEPSIZE="[350000]"
    ITERS=490000
    ANCHOR_SIZES="[64,128,56,512]"
    RATIOS="[0.5,1,2]"
    ANCHOR_STRIDES="[16,]"
    P4=True
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=fr-rcnn-weights/P4/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=fr-rcnn-weights/P4/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [ ! -f ${NET_FINAL}.index ]; then
  if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net_fr.py \
      --weight data/imagenet_weights/${NET}.ckpt \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --tag ${EXTRA_ARGS_SLUG} \
      --net ${NET} \
      --set ANCHOR_SIZES ${ANCHOR_SIZES} ANCHOR_RATIOS ${RATIOS} ANCHOR_STRIDES ${ANCHOR_STRIDES}\
      TRAIN.STEPSIZE ${STEPSIZE}  P4 ${P4} EXP_DIR P4/${NET} ${EXTRA_ARGS}
  else
    CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/trainval_net_fr.py \
      --weight data/imagenet_weights/${NET}.ckpt \
      --imdb ${TRAIN_IMDB} \
      --imdbval ${TEST_IMDB} \
      --iters ${ITERS} \
      --cfg experiments/cfgs/${NET}.yml \
      --net ${NET} \
      --set ANCHOR_SIZES ${ANCHOR_SIZES} ANCHOR_RATIOS ${RATIOS} ANCHOR_STRIDES ${ANCHOR_STRIDES}\
      TRAIN.STEPSIZE ${STEPSIZE} EXP_DIR P4/${NET}  P4 ${P4} ${EXTRA_ARGS}
  fi
fi

./experiments/scripts/test_P4.sh $@
