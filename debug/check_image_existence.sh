IMG_PATH="ImageNet/train/n01728572/n01728572_5245.JPEG"

if [ -f "$IMG_PATH" ]; then
  echo "$IMG_PATH : image exist"
else
  echo "$IMG_PATH : image not exist"
fi