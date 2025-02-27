# WZDNAS: Warm-up Strategies for Zero-Cost Differentiable NAS in Hardware Application Optimization
## WZDNAS(speed-up)
For SuperNet training during the warm-up phase, we decompose standard 3x3 convolutions into depth-wise separable convolutions to speed up the warm-up initialization. After warm-up phase, we combine their weights back into the original 3x3 convolution. By this method, we can significantly reduce SuperNet training time while maintaining the original architecture
for the search phase at no additional cost. However, this speed-up warm-up initialization
reduces the number of trainable parameters compared to the normal warm-up process, leading to a slight performance drop.

## Search comment
 - WZDNAS(speed-up) searching (with warm-up phase and hardware-aware loss)
```
python ./tools/train_loraAB_L1.py --cfg config/search/izdnasV4-P5-S42.yaml --data ./config/dataset/voc_dnas.yaml --hyp ./config/training/hyp.zerocost.yaml --model config/model/Search-YOLOv4-P5-SomeLayer.yaml --device 1 --exp_name EXP_NAME --nas DNAS-70 --zc naswot --lookup config/lookup/p5_rb5_gpu_overhead.yaml
```