## RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers

Below are instructions for reproducing the detection results of RepQ-ViT.
This repository is adopted from [*mmdetection*](https://github.com/open-mmlab/mmdetection) repo.

## Preliminaries

- Install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```bash
pip install -U openmim
mim install mmcv-full
```

- Install MMDetection.

```bash
cd RepQ-Vit/detection
pip install -v -e .
```

- Download pre-trained models from [Swin-Transformer-Object-Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) 

## Evaluation

- You can quantize and evaluate a single model using the following command:

```bash
python tools/test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval bbox segm [--w_bits] [--a_bits]

Required arguments:
 <CONFIG_FILE> : Path to config. You can find it at ./configs/swin/
 <DET_CHECKPOINT_FILE> : Path to checkpoint of pre-trained models.

optional arguments:
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
```

- Example: Quantize *Cascade Mask R-CNN with Swin-T* at W4/A4 precision:

```bash
python tools/test.py configs/swin/cascade_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_giou_4conv1f_adamw_3x_coco.py cascade_mask_rcnn_swin_tiny_patch4_window7.pth  --eval bbox segm
```

## Results

Below are the experimental results of our proposed RepQ-ViT that you should get on COCO dataset.

| Model                                     | Prec. | AP<sup>box</sup> / AP<sup>mask</sup> |
|:-----------------------------------------:|:-----:|:------------------------------------:|
| Mask RCNN w. Swin_T (46.0 / 41.6)         | W4/A4 | 36.1 / 36.0                          |
|                                           | W6/A6 | 45.1 / 41.2                          |
| Mask RCNN w. Swin_S (48.5 / 43.3)         | W4/A4 | 44.2 / 40.2                          |
|                                           | W6/A6 | 47.8 / 43.0                          |
| Cascade Mask RCNN w. Swin_T (50.4 / 43.7) | W4/A4 | 47.0 / 41.4                          |
|                                           | W6/A6 | 50.0 / 43.5                          |
| Cascade Mask RCNN w. Swin_S (51.9 / 45.0) | W4/A4 | 49.3 / 43.1                          |
|                                           | W6/A6 | 51.4 / 44.6                          |

## Citation

We appreciate it if you would please cite the following paper if you found the implementation useful for your work:

```bash
@inproceedings{li2023repq,
  title={Repq-vit: Scale reparameterization for post-training quantization of vision transformers},
  author={Li, Zhikai and Xiao, Junrui and Yang, Lianwei and Gu, Qingyi},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17227--17236},
  year={2023}
}
```
