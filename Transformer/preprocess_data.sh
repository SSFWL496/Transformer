path=$1
data_tag=$2
src=en
tgt=de
mkdir data-bin/$data_tag
output=data-bin/$data_tag
fairseq-preprocess --joined-dictionary --source-lang $src \
  --target-lang $tgt \
  --trainpref $path/train \
  --validpref $path/valid \
  --testpref $path/test  \
  --destdir $output \
  --workers 64
