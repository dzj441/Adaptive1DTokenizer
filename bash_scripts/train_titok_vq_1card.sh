CUDA_VISIBLE_DEVICES=0\
# WANDB_API_KEY=0022f2e7631e0264b23083ff83e6d0ca32ebb89e\
# WANDB_MODE=online\
# WANDB_PROJECT=titok_bl128_vq_debug\
# WANDB_NAME=titok_bl128_vq_debug_nocluster\ 
accelerate launch \
  --num_machines=1 \
  --num_processes=1 \
  --machine_rank=0 scripts/train_titok.py config=configs/training/adaptive1DTokenzier/titok_bl128_vq_noprior_debug.yaml \
    experiment.project="titok_bl128_vq_debug" \
    experiment.name="titok_bl128_vq_debug_nocluster_test" \
    experiment.output_dir="titok_bl128_vq_debug_nocluster_test" \

