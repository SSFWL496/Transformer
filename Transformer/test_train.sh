
export CUDA_VISIBLE_DEVICES=0

fairseq-train /home/wangchenglong/wngt/fairseq/data-bin/base_en_de_student_v1/ --arch transformer_wmt_en_de_tiny_6_1 \
    --share-all-embeddings \
    --fp16  \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --save-dir ./checkpoints/test \
    --max-update 300000 --save-interval-updates 5000 \
    --keep-interval-updates 40 \
    --encoder-normalize-before --decoder-normalize-before \
    --lr 7e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 200 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --update-freq 2 --no-epoch-checkpoints \
    --difficult-queue-size 30000
