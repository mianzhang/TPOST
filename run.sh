# HMM
# python main.py --dataset=small --model=HMM
# LM
# python main.py --dataset=small --epoch=30 --average --model=LM
# python main.py --dataset=large --epoch=10 --average --model=LM
# LogLM
python main.py --dataset=small --epoch=30 --lr=0.5 --l2_alpha=0.01 --batch_size=50 --model=LogLM --average
# python main.py --dataset=large --epoch=10 --lr=0.2 --l2_alpha=0.01 --batch_size=64 --model=LogLM
# GLLM
# python main.py --dataset=small --epoch=30 --model=GLM --average
# CRF
# python main.py --dataset=small --epoch=10 --lr=0.5 --l2_alpha=0.00 --batch_size=1 --model=CRF
# python main.py --dataset=large --epoch=10 --lr=0.2 --l2_alpha=0.01 --batch_size=64 --model=CRF
