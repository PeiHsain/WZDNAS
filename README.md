# ZeroDNAS
ZeroDNAS algorithm code for Qualcomm project

## File Structure
----
```
ZeroDNAS
└── data
    ├── coco  (Link)
    └── VOC2007 (Link) 

```

# Detail
- Search
    - IZeroDNAS
    - ZeroDNAS
    - DMaskingNAS
    - Zero-Cost EA
- Train
    - ZeroDNAS: 416x416
    - DNAS: 416x416
----

# Execution Step (For ScaledYOLOv4)
## 1 Search Command
### Train by ZeroDNAS
python ./tools/train_zdnas.py --cfg config/search/train_zdnas.yaml --data ./config/dataset/voc_dnas.yaml --hyp ./config/training/hyp.zerocost.yaml --model config/model/Search-YOLOv4-CSP.yaml --device 6 --exp_name EXP_NAME --nas DNAS-50 --zc naswot

#### Train by IZeroDNAS
python ./tools/train_izdnas.py --cfg config/search/train_izdnas.yaml --data ./config/dataset/voc_dnas.yaml --hyp ./config/training/hyp.zerocost.yaml --model config/model/Search-YOLOv4-CSP.yaml --device 6 --exp_name EXP_NAME --nas DNAS-50 --zc naswot

### Train by ZeroDNAS with the depth loss
python tools/train_zdnas_depth.py --cfg config/search/train_zdnas_416.yaml --data ./config/dataset/voc_dnas.yaml --hyp ./config/training/hyp.scratch.yaml --model ./config/model/Search-YOLOv4-P5-exp.yaml --device 6 --exp_name p5-voc-depth-50-seed44-fixgamma-0104 --nas DNAS-50 --zc naswot

### Train by ZeroDNAS with the look-up table
python tools/train_zdnas_table.py --cfg config/search/train_zdnas_416_coco.yaml --data ./config/dataset/coco_dnas.yaml --hyp ./config/training/hyp.scratch.yaml --model ./config/model/Search-YOLOv4-P5.yaml --device 5 --exp_name p5-coco-table-01-50-seed44-0107 --nas DNAS-50 --zc naswot --lookup config/lookup/p5_rb5_gpu.yaml

python tools/train_zdnas_table.py --cfg config/search/train_zdnas_416_coco.yaml --data ./config/dataset/coco_dnas.yaml --hyp ./config/training/hyp.scratch.yaml --model ./config/model/Search-YOLOv4-P5.yaml --device 4 --exp_name p5-coco-overhead-01-40-seed44-0107 --nas DNAS-40 --zc naswot --lookup config/lookup/p5_rb5_gpu_overhead.yaml

### To Be Continue ...

## 2 Train the model
please use the code of https://github.com/B106Roger/ScaledYOLOv4-NAS.git

### Training
python train.py --batch-size 32 --img-size 416 416 --data ./data/voc.yaml --cfg SEARCHED_ARCHITECTURE.yaml --weights '' --device 6 --name EXP_NAME --hyp ./data/hyp.finetune.yaml

- cfg: model config. should be something like (yolov4-p5.yaml, yolov4-csp.yaml, yolov4-csp-search.yaml)

### Testing
python test.py --img 416 --conf 0.001 --batch 8 --device 5 --data voc.yaml --weights BEST_WEIGHT.pt