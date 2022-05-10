def cdiffs-t5:
	python main.py \
    --task 'cdiffs' \
    --split test-full \
    --data_dir data \
    --out_dir out \
    --t5 ${MODEL} \
    --method direct \
    --prompt_tune \
    --do_train \
    --batch_size 32 \
    --lr 0.01

def cdiffw-t5:
	python main.py \
    --task 'cdiffw' \
    --split test-full \
    --data_dir data \
    --out_dir out \
    --t5 ${MODEL} \
    --method direct \
    --prompt_tune \
    --do_train \
    --batch_size 32 \
    --lr 0.01

def debug:
	python main.py \
    --task 'cdiffs' \
    --split dev \
    --data_dir data/debug \
    --out_dir out \
    --t5 google/t5-small-lm-adapt \
    --method direct \
    --prompt_tune \
    --do_train \
    --batch_size 32 \
    --lr 0.01