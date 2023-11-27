## RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers

Below are instructions for reproducing the classification results of RepQ-ViT.

## Evaluation

- You can quantize and evaluate a single model using the following command:

```bash
python test_quant.py [--model] [--dataset] [--w_bit] [--a_bit]

optional arguments:
--model: Model architecture, the choises can be: 
    vit_small, vit_base, deit_tiny, deit_small, deit_base, swin_tiny, swin_small.
--dataset: Path to ImageNet dataset.
--w_bit: Bit-precision of weights, default=4.
--a_bit: Bit-precision of activation, default=4.
```

- Example: Quantize *DeiT-S* at W4/A4 precision:

```bash
python test_quant.py --model deit_small --dataset <YOUR_DATA_DIR>
```

## Results

Below are the experimental results of our proposed RepQ-ViT that you should get on ImageNet dataset.

| Model          | Prec. | Top-1(%) | Prec. | Top-1(%) |
|:--------------:|:-----:|:--------:|:-----:|:--------:|
| ViT-S (81.39)  | W4/A4 | 65.05    | W6/A6 | 80.43    |
| ViT-B (84.54)  | W4/A4 | 68.48    | W6/A6 | 83.62    |
| DeiT-T (72.21) | W4/A4 | 57.43    | W6/A6 | 70.76    |
| DeiT-S (79.85) | W4/A4 | 69.03    | W6/A6 | 78.90    |
| DeiT-B (81.80) | W4/A4 | 75.61    | W6/A6 | 81.27    |
| Swin-T (81.35) | W4/A4 | 72.31    | W6/A6 | 80.69    |
| Swin-S (83.23) | W4/A4 | 79.45    | W6/A6 | 82.79    |

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
