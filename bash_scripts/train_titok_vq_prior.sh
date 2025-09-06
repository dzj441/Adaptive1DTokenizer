CUDA_VISIBLE_DEVICES=0 WANDB_MODE=online accelerate launch --dynamo_backend no --num_machines=1 --num_processes=1 --machine_rank=0 scripts/train_titok.py config=configs/training/adaptive1DTokenzier/titok_bl32_vq.yaml \
    experiment.project="titok_bl32_vq" \
    experiment.name="titok_bl32_vq_run1" \
    experiment.output_dir="titok_bl32_vq_run1_128" \