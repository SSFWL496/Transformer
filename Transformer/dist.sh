output_dir=./checkpoints/iwslt-de2en-tiny/student/
teacher_ckpt=./checkpoints/pre-iwlst-de-en/transformer_iwlst14de2en/checkpoint_best.pt
data_dir=./data-bin/iwslt14de2en/  #二进制数据文件
model_arch=transformer_wmt_en_de_tiny_6_1  #模型的arch文件 
distil_strategy=distil_all  # batch_level, selection, Word CE
# distil_strategy=global_level    #not use distillation
disitl_rate=0.5
queue_size=30000
 

export CUDA_VISIBLE_DEVICES=0,1,2,3

nohup fairseq-train $data_dir --arch $model_arch \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir $output_dir \
    --keep-last-epochs 7 \
    --save-interval-updates 3 \
    --max-epoch 60 \
    --fp16 \
    --lr 0.0007 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 1500 \
    --best-checkpoint-metric loss \
    --update-freq 8 --no-epoch-checkpoints \
    --use-distillation --teacher-ckpt-path $teacher_ckpt  --distil-strategy $distil_strategy --distil-rate $disitl_rate \
    --difficult-queue-size $queue_size \
    > dis3.log 2>&1 &
