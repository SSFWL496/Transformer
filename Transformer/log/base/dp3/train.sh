#! /usr/bin/bash
set -e

# device=0,1,2,3
#device=4,5,6
device=0,1,2,3


task=base-iwlst-de-en    #训练的模型种类
tag=transformer_iwlst14de2en   #模型标签
data_tag=iwslt14de2en #数据标签


# must set this tag
if [ $task == "base-iwlst-en-de" ]; then
        arch=transformer_iwslt_de_en
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=7e-4
        warmup=6000
        max_tokens=4096
        update_freq=2
        weight_decay=0.0001
        keep_last_epochs=20
        max_epoch=100
        max_update=
        dropout=0.3
        attention_dropout=0.1
        activation_dropout=0.1
        data_dir=$data_tag
        src_lang=en
        tgt_lang=de
        seed=2128977
elif [ $task == "base-iwlst-de-en" ]; then
        arch=transformer_iwslt_de_en
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=7e-4
        warmup=6000
        max_tokens=4096
        update_freq=2
        weight_decay=0.0001
        keep_last_epochs=20
        max_epoch=100
        max_update=
        dropout=0.3 
        attention_dropout=0.1
        activation_dropout=0.1
        data_dir=$data_tag
        src_lang=de
        tgt_lang=en
        seed=2128977
elif [ $task == "base-wmt14-en-de" ]; then
        arch=transformer_wmt_en_de
        share_embedding=1
        share_decoder_input_output_embed=0
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=7e-4
        warmup=8000
        max_tokens=4096
        update_freq=2
        weight_decay=0
        keep_last_epochs=20
        max_epoch=100
        max_update=
        dropout=0.1 
        attention_dropout=0.1
        activation_dropout=0.1
        data_dir=$data_tag
        src_lang=en
        tgt_lang=de
        seed=97926458
elif [ $task == "deep-wmt14-en-de" ]; then
        arch=transformer_wmt_en_de_deep48
        #arch=relative_transformer_t2t_wmt_en_de
        #arch=dense_relative_transformer_t2t_wmt_en_de
        #arch=relative_transformer_t2t_wmt_en_de_768
        #arch=transformer_wmt_en_de_big_t2t
        #arch=relative_transformer_t2t_wmt_en_de
        #arch=stochastic_relative_transformer_t2t_wmt_en_de
        share_embedding=1
        share_decoder_input_output_embed=0
        #深层设为1
        criterion=label_smoothed_cross_entropy
        fp16=1
        lr=0.002
        #比较宽0.0016
        warmup=16000
        #16000
        max_tokens=2048
        #2048
        update_freq=16
        #2
        weight_decay=0.0
        keep_last_epochs=5
        max_epoch=21
        max_update=
        data_dir=$data_tag
        src_lang=en
        tgt_lang=de
else
        echo "unknown task=$task"
        exit
fi

save_dir=checkpoints/$task/$tag

if [ ! -d $save_dir ]; then
        mkdir -p $save_dir
fi
cp ${BASH_SOURCE[0]} $save_dir/train.sh

gpu_num=`echo "$device" | awk '{split($0,arr,",");print length(arr)}'`

cmd="fairseq-train data-bin/$data_dir
  --distributed-world-size $gpu_num -s $src_lang -t $tgt_lang
  --arch $arch
  --optimizer adam --clip-norm 0.0
  --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $warmup
  --lr $lr --min-lr 1e-09
  --weight-decay $weight_decay
  --criterion $criterion  --label-smoothing 0.1
  --max-tokens $max_tokens
  --update-freq $update_freq
  --no-progress-bar
  --log-interval 100
  --ddp-backend no_c10d
  --log-format tqdm
  --log-interval 100
  --save-dir $save_dir
  --keep-last-epochs $keep_last_epochs "


  #--decoder-embed-path /home/zhangyuhao/pretrain.out
adam_betas="'(0.9, 0.98)'"
cmd=${cmd}" --adam-betas "${adam_betas}
if [ $share_embedding -eq 1 ]; then
cmd=${cmd}" --share-all-embeddings "
fi
if [ $share_decoder_input_output_embed -eq 1 ]; then
cmd=${cmd}" --share-decoder-input-output-embed "
fi
if [ -n "$max_epoch" ]; then
cmd=${cmd}" --max-epoch "${max_epoch}
fi
if [ -n "$max_update" ]; then
cmd=${cmd}" --max-update "${max_update}
fi
if [ -n "$dropout" ]; then
cmd=${cmd}" --dropout "${dropout}
fi
if [ $fp16 -eq 1 ]; then
cmd=${cmd}" --fp16 "
fi
if [ -n "$seed" ]; then
cmd=${cmd}" --seed "${seed}
fi
if [ -n "$dropout" ]; then
cmd=${cmd}" --dropout "${dropout}
fi
if [ -n "$attention_dropout" ]; then
cmd=${cmd}" --attention-dropout "${attention_dropout}
fi
if [ -n "$activation_dropout" ]; then
cmd=${cmd}" --activation-dropout "${activation_dropout}
fi

#echo $cmd
#eval $cmd
#cmd=$(eval $cmd)
#nohup $cmd exec 1> $save_dir/train.log exec 2>&1 &
#tail -f $save_dir/train.log

export CUDA_VISIBLE_DEVICES=$device
cmd="nohup "${cmd}" > $save_dir/train.log 2>&1 &"
eval $cmd
tail -f $save_dir/train.log
