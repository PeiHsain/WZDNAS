# File Architecture
```
lib
└── core: warm-up and searching functions
|    ├── izdnas_all.py: 
|    |     ```train_epoch_dnas```: Warm-up function
|    |     ```train_epoch_zdnas_all```: architecture search function
|    └── ....
└── models: architectures of YOLO model
└── utils: other functions 
└── zero_proxy: zero-cost proxies
```