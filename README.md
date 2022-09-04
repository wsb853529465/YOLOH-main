# YOLOH

This is a PyTorch version of YOLOH.

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yoloh python=3.6
```

- Then, activate the environment:
```Shell
conda activate yoloh
```

- Requirements:
```Shell
pip install -r requirements.txt 
```

We suggest that PyTorch should be higher than 1.9.0 and Torchvision should be higher than 0.10.3. At least, please make sure your torch is version 1.x.

# Main results on COCO-val

| Model                 | scale    | mAP  |
| --------------------- | -------- | ---- |
| YOLOH_R_50_DC5_640_1x | 640, 640 | 39.8 |

Due to device reasons, larger backbones require insufficient video memory to support batchsize = 16. So the bigger backbone has not been written yet。

# Train

## Single GPU
```Shell
sh train.sh
```

You can change the configurations of `train.sh`, according to your own situation.

### or

```Shell
python train.py --cuda -d coco --root dataset/ -v yoloh50-DC5-640 -lr 0.03 -lr_bk 0.01 --batch_size 16 --train_min_size 640 --train_max_size 640 --val_min_size 640 --val_max_size 640 --schedule 1x --grad_clip_norm 4.0 
```



# Test
```Shell
python test.py -d coco \
               --cuda \
               -v yoloh50 \
               --weight path/to/weight \
               --min_size 800 \
               --max_size 1333 \
               --root path/to/dataset/ \
               --show
```

# Demo
```Shell
python demo.py --mode image \
               --path_to_img data/demo/images/ \
               -v yoloh50 \
               --cuda \
               --weight path/to/weight \
               --min_size 800 \
               --max_size 1333 \
               --show
```

