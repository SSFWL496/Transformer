#!/usr/bin/bash
set -e

dev=$1
data_dir=$2
model_dir=$3
# set tag
checkpoint=$4
beam=$5
lenpen=$6
batch_size=$7
who=$8
record=$9
bpe_pattern=${10}
src=${11}
tar=${12}
#evalset=$10

if [ $checkpoint = "model_ensemble" ]; then
  output_dir=../ensemble/
  model=$model_dir
else
  output_dir=$model_dir
  model=$model_dir/$checkpoint
fi

output=$output_dir/translation.$who.beam$beam.$lenpen.device$dev.log
CUDA_VISIBLE_DEVICES=$dev fairseq-generate \
$data_dir \
--path $model \
--gen-subset $who \
--batch-size $batch_size \
--quiet \
--beam $beam \
--lenpen $lenpen \
--skip-invalid-size-inputs-valid-test \
--remove-bpe $bpe_pattern > $output

#--sampling \
# python3 rerank.py $output_dir/hypo.$who.beam$beam.lenpen$lenpen.device$dev $output_dir/hypo.$who.beam$beam.lenpen$lenpen.device$dev.decodes
#remove the intermediate output
#rm $model_dir/hypo.$who.beam$beam.lenpen$lenpen

# BLEU=`tail -1 $output |cut -d " " -f 8| awk '{$a=substr($0,0,length($0)-1);print $a;}'`
#echo $BLEU
# BP=`tail -1 $output |cut -d " " -f 10| awk '{$a=substr($0,5,length($0)-5);print $a;}'`
# #echo $BP
# RATIO=`tail -1 $output |cut -d " " -f 11| awk '{$a=substr($0,7,length($0)-7);print $a;}'`
#echo $RATIO

BLEU=`tail -1 $output`

#echo -e "Set=$evalset\twho=$who\tAlpha=$lenpen\tBeam=$beam\tBatch=$batch_size\tBLEU=$BLEU\tBP=$BP\tRatio=$RATIO\n" >> $record
echo -e "who=$who\tAlpha=$lenpen\t\tBLEU=$BLEU\n" >> $record

