def cdiffs-gpt:
	python main.py \
    --task 'cdiffs' \
    --split dev \
    --data_dir data \
    --out_dir out \
    --gpt2 gpt2-small \
    --method direct \
    --prompt_tune \
    --do_train \
    --batch_size 32 \
    --lr 0.01

def cdiffw-gpt:
	python main.py \
    --task 'cdiffw' \
    --split dev \
    --data_dir data \
    --out_dir out \
    --gpt2 gpt2-small \
    --method direct \
    --prompt_tune \
    --do_train \
    --batch_size 32 \
    --lr 0.01