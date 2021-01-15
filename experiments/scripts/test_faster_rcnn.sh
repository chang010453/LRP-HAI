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
    ITERS=70000
    ANCHOR_SIZES="[128,256,512]"
    RATIOS="[0.5,1,2]"
    ANCHOR_STRIDES="[16,]"
    P4=False
    ;;
  pascal_voc_0712)
    TRAIN_IMDB="voc_2007_trainval+voc_2012_trainval"
    TEST_IMDB="voc_2007_test"
    ITERS=180000
    ANCHOR_SIZES="[128,256,512]"
    RATIOS="[0.5,1,2]"
    ANCHOR_STRIDES="[16,]"
    P4=False
    ;;
 cell)
    TRAIN_IMDB="cell_train"
    TEST_IMDB="cell_val"
    ITERS=110000
    ANCHOR_SIZES="[64,128,256,512]"
    RATIOS="[0.5,1,2]"
    ANCHOR_STRIDES="[16,]"
    P4=False
	;;
  coco)
    TRAIN_IMDB="coco_2014_train+coco_2014_valminusminival"
    TEST_IMDB="coco_2014_minival"
    ITERS=490000
    ANCHOR_SIZES="[64,128,256,512]"
    RATIOS="[0.5,1,2]"
    ANCHOR_STRIDES="[16,]"
    P4=False
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/test_${NET}_${TRAIN_IMDB}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

set +x
if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  NET_FINAL=fr-rcnn-weights/${NET}/${TRAIN_IMDB}/${EXTRA_ARGS_SLUG}/${NET}_faster_rcnn_iter_${ITERS}.ckpt
else
  NET_FINAL=fr-rcnn-weights/${NET}/${TRAIN_IMDB}/default/${NET}_faster_rcnn_iter_${ITERS}.ckpt
fi
set -x

if [[ ! -z  ${EXTRA_ARGS_SLUG}  ]]; then
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net_fr.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --tag ${EXTRA_ARGS_SLUG} \
    --net ${NET} \
    --set ANCHOR_SIZES ${ANCHOR_SIZES} ANCHOR_RATIOS ${RATIOS} ANCHOR_STRIDES ${ANCHOR_STRIDES} P4 ${P4}\
          ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${GPU_ID} time python ./tools/test_net_fr.py \
    --imdb ${TEST_IMDB} \
    --model ${NET_FINAL} \
    --cfg experiments/cfgs/${NET}.yml \
    --net ${NET} \
    --set ANCHOR_SIZES ${ANCHOR_SIZES} ANCHOR_RATIOS ${RATIOS} ANCHOR_STRIDES ${ANCHOR_STRIDES} P4 ${P4}\
          ${EXTRA_ARGS}
fi

