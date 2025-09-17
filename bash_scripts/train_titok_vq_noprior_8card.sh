CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\
WANDB_MODE=online\
WANDB_PROJECT=titok_bl32_vq\
WANDB_NAME=titok_bl32_vq_8card_noPrior\ 
accelerate launch \
  --num_machines=1 \
  --num_processes=8 \
  --machine_rank=0 scripts/train_titok.py config=configs/training/adaptive1DTokenzier/titok_bl32_vq_local.yaml \
    experiment.project="titok_bl32_vq" \
    experiment.name="titok_bl32_vq_128_8card_noPrior" \
    experiment.output_dir="titok_bl32_vq_8card_noPrior_wholeset" \