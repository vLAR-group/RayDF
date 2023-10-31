CUDA_VISIBLE_DEVICES=${1} python run_cls.py --config configs/${2}_cls.txt --scene ${3}
CUDA_VISIBLE_DEVICES=${1} python run_mv.py --config configs/${2}.txt --scene ${3}
CUDA_VISIBLE_DEVICES=${1} python run_mv.py --config configs/${2}.txt --scene ${3} --rgb_layer 2