CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
WANDB_API_KEY=0022f2e7631e0264b23083ff83e6d0ca32ebb89e\
WANDB_MODE=online\
WANDB_PROJECT=titok_bl128_vq\
WANDB_NAME=titok_bl128_vq_8card_Prior\ 
accelerate launch \
  --num_machines=1 \
  --num_processes=8 \
  --machine_rank=0 scripts/train_titok.py config=configs/training/adaptive1DTokenzier/titok_bl128_vq_prior.yaml \
    experiment.project="titok_bl128_vq" \
    experiment.name="titok_bl128_vq_8card_Prior" \
    experiment.output_dir="titok_bl128_vq_8card_Prior" \

