export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --main_process_port=2950 selection/iter.py
