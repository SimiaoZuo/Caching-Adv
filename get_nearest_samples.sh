CUDA_VISIBLE_DEVICES=0 python nearest_samples.py \
    ./data-bin/iwslt14.tokenized.de-en \
    --path $1 \
    --batch-size 256 \
    --gen-subset 'train' \
    --n-largest 10 \
    --prop 0.1 \
    --save-name 'iwslt14.tokenized.de-en.neighbors' \
    --seed 1 \
