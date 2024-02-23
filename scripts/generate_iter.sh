export CUDA_VISIBLE_DEVICES=0,1

accelerate launch --main_process_port=2950 selection/iter.py --selection/indices/cohere_KMeansIter_0.1_Llama-2-7b-hf_round_4_iter_1.pkl
