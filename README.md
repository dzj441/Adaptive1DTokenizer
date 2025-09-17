## configs
training scripts are at "./bash_scripts"
training configs are at "./configs"
## create environment
```
bash bash_scripts/create_env.sh
```
## dataset
recommend using symbolic link  
dataset format should be
```
path/to/trainset
                /n01440764
                        /many_pics.JPEG
                /n01443537
                        /many_pics.JPEG
                ......
path/to/valset
                /n01440764
                        /many_pics.JPEG
                /n01443537
                        /many_pics.JPEG
                ......
```
## training
train a no prior loss version  
```
bash bash_scripts/train_titok_vq_noprior_8card.sh
```
train a prior loss version  
```
bash bash_scripts/train_titok_vq_prior_8card.sh
```