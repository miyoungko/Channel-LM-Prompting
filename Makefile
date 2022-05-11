def cdiffs-t5:
	python main.py \
    --task 'cdiffs' \
    --split dev \
    --test_split test \
    --data_dir data/original/cdiffs \
    --out_dir out \
    --t5 ${MODEL}  \
    --method direct \
    --prompt_tune \
    --do_train \
    --n_prefix 100 \
    --batch_size 16 \
    --lr 0.3

def cdiffw-t5:
	python main.py \
    --task 'cdiffw' \
    --split dev \
    --test_split test \
    --data_dir data/original/cdiffw \
    --out_dir out \
    --t5 ${MODEL}  \
    --method direct \
    --prompt_tune \
    --do_train \
    --n_prefix 100 \
    --batch_size 16 \
    --weight \
    --lr 0.3

def debug:
	python main.py \
    --task 'cdiffs' \
    --split dev \
    --test_split test \
    --data_dir data/debug \
    --out_dir out \
    --t5 google/t5-small-lm-adapt \
    --method direct \
    --prompt_tune \
    --do_train \
    --batch_size 32 \
    --lr 0.01