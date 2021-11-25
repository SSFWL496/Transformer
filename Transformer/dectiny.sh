#! /usr/bin/bash
set -e

# device, you can set multiple devices, e.g. device=(0 1 2)
# then program will parallelly translate over various evalset (e.g. evalset=(cwmt18-dev mt06 mt08), or over various alpha (e.g. alphas=(1.0 1.1 1.2). 
# However, note that multiple evalset and multiple alpha can not set concurrently.
# more device will not be used. e.g. you set device=(0 1 2 3), but you only choose three evalset, the gpu=3 will not be used
device=(3)
#device=(4 5 6 7)
#device=(0 1 2 3)
# evaluation or just decoding
is_eval=1
# choice in 'multi-bleu' and 'mteval'
eval_tool='multi-bleu'
# only work for 'multi-bleu'
lowercase=0

######## dataset ########
task=tiny-iwlst-de-en
## *****dataset, set it!*****
who=(test)
#who=(test)
####### model ########
# your tag, must set!
model_dir_tag=transformer_iwlst14de2en #模型tag
#checkpoint_name=checkpoint_temp.pt        #模型的名称
checkpoint_name=last5.ensemble.pt
data_tag=iwslt14de2en  #数据tag
#model_dir_tag=iu2en-final-dense-45-8-pesudo-ftbase312
#model_dir_tag=iu2en-final-big-pesudo-ftbase312
#model_dir_tag=iu2en-final-nrbt1-40-8-pesudo-finetune
#model_dir_tag=iu2en-final-pretrain-pesudo-ftbase3124
#model_dir_tag=iu2en-final-768-16-8-pesudo-ftbase312
if [ $task == "deep-wmt14-en-de" ]; then
        data_dir=$data_tag
        checkpoint_epoch=$checkpoint_name
        batch_size=64
        beam_size=4
        ensemble=5 
        #length_penalty=(2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6)
        length_penalty=(0.6)
        src_lang=en 
        tgt_lang=de
        sacrebleu_set=
elif [ $task == "base-iwlst-de-en" ]; then
        data_dir=$data_tag
        checkpoint_epoch=$checkpoint_name
        batch_size=64
        beam_size=4
        ensemble=5
        #length_penalty=(2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6)
        length_penalty=(0.6)
        src_lang=de
        tgt_lang=en
        sacrebleu_set=
elif [ $task == "tiny-iwlst-de-en" ]; then
        data_dir=$data_tag
        checkpoint_epoch=$checkpoint_name
        batch_size=64
        beam_size=4
        ensemble=5
        #length_penalty=(2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6)
        length_penalty=(0.6)
        src_lang=de
        tgt_lang=en
        sacrebleu_set=
elif [ $task == "base-iwlst-en-de" ]; then
        data_dir=$data_tag
        checkpoint_epoch=$checkpoint_name
        batch_size=64
        beam_size=5
        ensemble=5
        #length_penalty=(2.2 2.4 2.6 2.8 3.0 3.2 3.4 3.6)
        length_penalty=(1.1)
        src_lang=en
        tgt_lang=de
        sacrebleu_set=
fi
######## decoding ########
# you can semultiple alpha, e.g. tuning on dev set. usage: alpha=(0.9 1.0 1.1), split by space
#alphas=(0.9 1.0 1.1 1.2 1.3 1.4)
alphas=(${length_penalty[*]})
# used for specific model file
output_dir=./checkpoints/$task/$model_dir_tag
checkpoint=$checkpoint_epoch
# used for fairseq test file
eval_dir=./data-bin/$data_dir

# generate ensemble model if not exist. 
# program can generate average checkpoint automatically
if [ -n "$ensemble" ]; then
 	# use the first gpu to generate ensembled model
    if [ ! -e "$output_dir/last$ensemble.ensemble.pt" ]; then
        PYTHONPATH=`pwd` python3 scripts/average_checkpoints.py --inputs $output_dir --output $output_dir/last$ensemble.ensemble.pt --num-epoch-checkpoints $ensemble
    fi
    checkpoint=checkpoint_best.pt
    #checkpoint=last$ensemble.ensemble.pt
fi

#checkpoint=checkpoint_best.pt
# check illegal
n_who=${#who[@]}
n_alphas=${#alphas[@]}
n_device=${#device[@]}
if [ $n_who -gt 1 -a $n_alphas -gt 1 ]; then
	echo "evalset-who and alphas can not be multiple concurrently"
	exit
fi

# set ensemble=0 to avoid None
if [ -z "$ensemble" ]; then
	ensemble=0
fi

# generate random number for record
record=$(date +%s%N).tmp
 
# multiple alphas
if [ ${#who[@]} -eq 1 ]; then
	# device is enough
	if [ $n_device -ge $n_alphas ]; then
		#echo "device is enough!"
		for ((i=0;i<${#alphas[@]};i++));do
		{
			alpha=${alphas[$i]}
			dev=${device[$i]}
			echo "run alpha=$alpha dev=$dev"
				./translate.sh $dev $eval_dir $output_dir $checkpoint $beam_size $alpha $batch_size $who $record
		}&
		done
		#echo "[enough]==> wait it"
		wait
	# device is poor
	else
		#echo "device is poor"
		if [ $(($n_alphas%$n_device)) -eq 0 ]; then
			n_group=$(($n_alphas/$n_device))
		else
			n_group=$(($n_alphas/$n_device+1)) 
		fi
		#echo "group=$n_group"
		for ((i=0;i<$n_group;i++));do
		{
			for ((j=0;j<$n_device;j++));do
			{
				alpha=${alphas[$(($i*$n_device+$j))]}
				dev=${device[$(($j))]}
				if [ -n "$alpha" ]; then
					./translate.sh $dev $eval_dir $output_dir $checkpoint $beam_size $alpha $batch_size $who $record $src_lang $tgt_lang
					echo "dev=$dev alpha=$alpha finish"
				fi
			} &
			done
			#echo "wait group=$i finish"
			wait
			#echo "group=$i finish"
		}
		done
	fi


# multiple evalset
else
        # device is enough
        if [ $n_device -ge $n_who ]; then
                #echo "device is enough!"
                for ((i=0;i<${#who[@]};i++));do
                {
                        alpha=${alphas[0]}
                        dev=${device[$i]}
                        who=${who[$i]}
                        echo "run ${who[$i]} alpha=$alpha dev=$dev"
                        ./translate.sh $dev $eval_dir $output_dir $beam_size $alpha $batch_size $who $record  $src_lang $tgt_lang
                }&
                done
                #echo "[enough]==> wait it"
                wait
        # device is poor
        else
                #echo "device is poor"
                if [ $(($n_evalset%$n_device)) -eq 0 ]; then
                        n_group=$(($n_evalset/$n_device))
                else
                        n_group=$(($n_evalset/$n_device+1))
                fi
                #echo "group=$n_group"
                for ((i=0;i<$n_group;i++));do
                {
                        for ((j=0;j<$n_device;j++));do
                        {
                                alpha=${alphas[0]}
                                dev=${device[$(($j))]}
                                who=${who[$(($i*$n_device+$j))]}
                                if [ -n "$data" ]; then
                                        ./translate.sh $dev $eval_dir $output_dir $beam_size $alpha $batch_size $who $record $evalset $src_lang $tgt_lang
                                        echo "dev=$dev set=$who alpha=$alpha finish"
                                fi
                        } &
                        done
                        #echo "wait group=$i finish"
                        wait
                        #echo "group=$i finish"
                }
                done
        fi
fi




# todo: format print
echo "********************"
echo -e "Tag=$model_dir_tag\tEval=$eval_tool\tLowercase=$lowercase\n"
while read LINE
do
        echo -e $LINE
done  < $record
rm $record
