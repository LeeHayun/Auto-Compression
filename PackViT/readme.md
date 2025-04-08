# PackViT

## Usage

### Environment Settings

```
conda create -n packvit python=3.6

conda activate packvit

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch

pip3 install timm==0.4.5
```

### Data preparation 
download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

### Download prertained model for training 

```
sh download_pretrain.sh
```

### Training


**DeiT-S**

```
CUDA_VISIBLE_DEVICES="0,1,2,3"  python3 -u -m torch.distributed.launch --nproc_per_node=4 --use_env main_l2_vit_3keep_senet_mlerp.py 
                                --output_dir logs/3keep_senet_mlerp 
                                --arch deit_small 
                                --input-size 224 
                                --batch-size 256 
                                --data-path /data/ImageNet
                                --epochs 30 
                                --dist-eval 
                                --distill 
                                --base_rate 0.7 
```
**DeiT-B**

```
CUDA_VISIBLE_DEVICES="0,1,2,3"  python3 -u -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit_3keep_senet_mlerp.py 
                                --output_dir logs/deit_base_3keep_senet_mlerp_256_60_5e-4 
                                --arch deit_base 
                                --input-size 224 
                                --batch-size 256 
                                --data-path /data/ImageNet
                                --epochs 60 
                                --dist-eval 
                                --distill 
                                --base_rate 0.7 

```


### Inference

```
python infer.py --data-path /home/imagenet --model deit_small_mlerp --model-path checkpoint_best.pth --base_rate 0.7 
```


## Acknowledgements

Our code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit), [DynamicViT](https://github.com/raoyongming/DynamicViT), [SPViT](https://github.com/PeiyanFlying/SPViT).

