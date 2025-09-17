CUDA_VISIBLE_DEVICES=0\
WANDB_MODE=online\
WANDB_PROJECT=titok_bl32_vq\
WANDB_NAME=titok_bl32_vq_128_prior_XXS_lce0.01\ 
accelerate launch --num_machines=1 --num_processes=1 --machine_rank=0 scripts/train_titok.py config=configs/training/adaptive1DTokenzier/titok_bl32_vq.yaml \
    experiment.project="titok_bl32_vq" \
    experiment.name="titok_bl32_vq_run1" \
    experiment.output_dir="titok_bl32_vq_XXS_lce_0.01" \