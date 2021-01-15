# LRP-HAI from *Sequential Objection Detection Based on Deep Reinforcement Learning with a Hybrid Attention Interface*

Code is implemented on [faster-rcnn](https://github.com/endernewton/tf-faster-rcnn) and [drl-rpn](https://github.com/aleksispi/drl-rpn-tf)

Installation:
------
1.先安裝conda environment  
	 conda env create -f drl-rpn.yml  
 
2.參考 [faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)的installation, 但步驟稍微不同  
  1). 不需要 Clone the repository  
  2)和3): lib_fr是faster r-cnn的library, lib_drl是加入HAI的library, lib_fr 和 lib_drl 都要安裝, 這邊以lib_drl為例   
    Update your -arch in setup script to match your GPU  
    cd lib_drl  
    # Change the GPU architecture (-arch) if necessary  
    gedit setup.py 照他的方式改  
    make clean  
    make  
    cd ..  
  4). Install the Python COCO API  
    cd data/coco/PythonAPI  
    make  
    cd ../../..  

3.Setup data:   
  參考[faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)的setup data  

Configuration:
--------
如果想要調整超參數設定, 可以修改./experiments/cfgs/裡面的yml檔

Training:
----------
如果要重新訓練faster rcnn:  
  如要訓練 vgg16, res101  
    1). training: 執行 ./experiments/scripts/train_faster_rcnn.sh GPU_ID DATASET NET  
    2). testing: 執行 ./experiments/scripts/test_faster_rcnn.sh GPU_ID DATASET NET  
    測試結果位於 ./fr-rcnn-weights/${NET}/${TEST_IMDB}  
    example: 只須要選擇使用的網路跟GPU  
     ./experiments/scripts/train_faster_rcnn.sh 0 cell vgg16  
     ./experiments/scripts/test_faster_rcnn.sh 1 pascal_voc_0712 res101  
	 如要訓練 P4 ([NET]固定為res101)  
    1). training: 執行 ./experiments/scripts/train_P4.sh GPU_ID DATASET res101  
    2). testing: 執行 ./experiments/scripts/test_P4.sh GPU_ID DATASET res101  
    測試結果位於 ./fr-rcnn-weights/P4/res101/${TEST_IMDB}  
    example: 只須要選擇使用GPU, 其他都一樣  
     ./experiments/scripts/train_P4.sh 0 cell res101  
     ./experiments/scripts/test_P4.sh 1 cell res101  

訓練LRP-HAI:  
  alpha=True  
	 如feature extractor為vgg16, res101:  
    1). training:  
     設定SAVE_PATH(儲存位置), WEIGHTS_PATH(pretrained_model位置)  
     執行 ./experiments/scripts/train_LRP_HAI.sh GPU_ID DATASET NET DET_START ITERS ALPHA  
    2). testing:   
     設定WEIGHTS_PATH(訓練好的model位置)  
     執行 ./experiments/scripts/train_LRP_HAI.sh GPU_ID DATASET NET NBR_FIX ALPHA  
     測試結果位於 ./output/  
    example: 只須要選擇使用的網路跟GPU, 其他都一樣  
     ./experiments/scripts/train_LRP_HAI.sh 0 cell vgg16 40000 110000 True  
     ./experiments/scripts/test_LRP_HAI.sh 1 cell res101 0 True  
  如feature extractor為P4:  
    NET設為res101  
    1). training:   
     設定SAVE_PATH(儲存位置), WEIGHTS_PATH(pretrained_model位置)  
     執行 ./experiments/scripts/train_LRP_HAI_P4.sh GPU_ID DATASET NET DET_START ITERS ALPHA  
    2). testing:   
     設定WEIGHTS_PATH(訓練好的model位置)		 
     執行 ./experiments/scripts/test_LRP_HAI_P4.sh GPU_ID DATASET NET NBR_FIX ALPHA  
     測試結果位於 ./output/  
    example: 只須要選擇使用GPU, 其他都一樣  
     ./experiments/scripts/train_LRP_HAI_P4.sh 0 pascal_voc_0712 res101 40000 110000 True  
     ./experiments/scripts/test_LRP_HAI_P4.sh 1 pascal_voc_0712 res101 0 True  

DEMO:
----
執行./tool/demo_drl.py，如要調整使用的weights, config, datasets的話在parse_args()調  
記得要修改所使用的config.yml檔, DO_VISUALIZE改成True  
