def cdiffs-gpt:
	python main.py \
    --task 'cdiffs' \
    --split test \
    --data_dir data \
    --out_dir out \
    --gpt2 ${MODEL} \
    --method direct \
    --prompt_tune \
    --do_train \
    --batch_size 64 \
    --lr 0.01

def cdiffw-gpt:
	python main.py \
    --task 'cdiffw' \
    --split test \
    --data_dir data \
    --out_dir out \
    --gpt2 ${MODEL} \
    --method direct \
    --prompt_tune \
    --do_train \
    --batch_size 64 \
    --lr 0.01