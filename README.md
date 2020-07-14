# Voting4SC
## Contents
* [Introduction](#Introduction)
* [Prerequisites](#Prerequisites)
* [Training](#Training)
* [Inference](#Inference)
* [Citation](#Citation)


## Introduction
This is the code for the paper "[Modeling Voting for System Combination in Machine Translation](https://www.ijcai.org/Proceedings/2020/511)" (IJCAI 2020).  The implementation is on top of the open-source NMT toolkit [THUMT](https://github.com/thumt/THUMT). You might need to glance over the user manual of THUMT for knowing the basic usage of THUMT.

## Prerequisites
* Python 2.7
* Tensorflow 1.11 - 1.15

## Training
```
PYTHONPATH=${path_to_Voting4SC} \
python ${path_to_Voting4SC}/thumt/bin/trainer.py \
    --input ${train_src} ${train_trg} ${train_hyp1} ${train_hyp2} ${train_hyp3} \
    --vocabulary ${vocab_src} ${vocab_trg} \
    --model transformer \
    --validation ${dev_src} ${dev_hyp1} ${dev_hyp2} ${dev_hyp3} \
    --references ${dev_ref1} ${dev_ref2} ... \
    --parameters update_cycle=3,constant_batch_size=false,batch_size=1667,eval_steps=2000,eval_batch_size=12,train_steps=100000,save_checkpoint_steps=2000,keep_top_checkpoint_max=10,keep_checkpoint_max=9999999,\
device_list=[0,1,2,3,4],hidden_size=512,num_encoder_layers=6,num_decoder_layers=6,learning_rate=1,warmup_steps=4000,residual_dropout=0.1,relu_dropout=0.1,attention_dropout=0.1,num_heads=8,\
shared_embedding_and_softmax_weights=true,shared_source_target_embedding=true,buffer_size=10000000,layer_preprocess="layer_norm",layer_postprocess="none",position_info_type="absolute",sc_num=3
```
* hyperparam "sc_num": number of system to be combined during Training or Inference (defalt value = 3).

## Inference
```
PYTHONPATH=${path_to_Voting4SC} \
python ${path_to_Voting4SC}/thumt/bin/translator.py \
    --input ${src} ${hyp1} ${hyp2} ${hyp3} \
    --output ${trans} \
    --vocabulary ${vocab_src} ${vocab_trg} \
    --checkpoints ./train/eval \
    --model transformer \
    --parameters device_list=[0,1,2,3],decode_alpha=1.0,decode_batch_size=20,sc_num=3
```

## Citation
If you use our codes, please cite our paper:
```
@inproceedings{ijcai2020-511,
  title     = {Modeling Voting for System Combination in Machine Translation},
  author    = {Huang, Xuancheng and Zhang, Jiacheng and Tan, Zhixing and Wong, Derek F. and Luan, Huanbo and Xu, Jingfang and Sun, Maosong and Liu, Yang},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},             
  editor    = {Christian Bessiere}
  pages     = {3694--3701},
  year      = {2020},
  month     = {7},
  note      = {Main track}
  doi       = {10.24963/ijcai.2020/511},
  url       = {https://doi.org/10.24963/ijcai.2020/511},
}
```
