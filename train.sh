#!/usr/bin/env bash
save_model=./models2/model0
logs=./models2/log0.txt
lr=0.0001

CUDA_VISIBLE_DEVICES=0 python -u train_model.py --model_dir=${save_model} \
                                                --learning_rate=${lr} \
                                                --lr_epoch='20,50,100,300,400' \
				                				--level=L1 \
				                				--image_size=112 \
				                				--image_channels=3 \
				                				--batch_size=128 \
					        					--max_epoch=500 \
                                                > ${logs} 2>&1 &
tail -f ${logs}
