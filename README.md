# MQTransformer


This repo is the official implementation of ["MQTransformer"](https://arxiv.org/abs/2205.14354) as well as the follow-ups. It currently includes code and models for the following tasks:



## Updates


***02/07/2023***

`News`: 

1. The Thirty-Seventh Conference on Artificial Intelligence (AAAI2023) will be held in Washington, DC, USA., from February 7-14, 2023.


***04/04/2023***

`News`: 

1. We release the MQTransformer code. 


## Introduction

**MQTransformer** 
Previous multi-task dense prediction studies developed complex pipelines such as multi-modal distillations in multiple stages or searching for task relational contexts for each task. The core insight beyond these methods is to maximize the mutual effects of each task. Inspired by the recent query-based Transformers, we propose a simple pipeline named Multi-Query Transformer (MQTransformer) that is equipped with multiple queries from different tasks to facilitate the reasoning among multiple tasks and simplify the cross-task interaction pipeline. Instead of modeling the dense per-pixel context among different tasks, we seek a task-specific proxy to perform cross-task reasoning via multiple queries where each query encodes the task-related context. The MQTransformer is composed of three key components: shared encoder, cross-task query attention module and shared decoder. We first model each task with a task-relevant query. Then both the task-specific feature output by the feature extractor and the task-relevant query are fed into the shared encoder, thus encoding the task-relevant query from the task-specific feature. Secondly, we design a cross-task query attention module to reason the dependencies among multiple task-relevant queries; this enables the module to only focus on the query-level interaction. Finally, we use a shared decoder to gradually refine the image features with the reasoned query features from different tasks. Extensive experiment results on two dense prediction datasets (NYUD-v2 and PASCAL-Context) show that the proposed method is an effective approach and achieves state-of-the-art results. 

MQTransformer achieves strong performance on NYUD-v2 semantic segmentation (`54.84 mIoU` on test), surpassing previous models by a large margin.

![MQTransformer](figures/overview.png)
<p align="center"> An overview of MQTransformer. The MQTransformer represents multiple task-relevant queries to extract task-specific features from different tasks and performs joint multi-task learning. Here, we show an example of task-specific policy learned using our method. Note that the encoder (aqua) and decoder (mauve) are shared in our model. The number of task queries depends on the number of tasks. There are $T$ tasks. The task query first generates uniform initialization weights and then applies these task queries to encode from the corresponding task-specific feature in the shared encoder. The 'Seg', 'Part' and 'Normals' mean semantic segmentation and human part segmentation and surface normals tasks, respectively.</p>


![MQTransformer](figures/encoder-decoder.png)
<p align="center">Illustration of the Multi-Query Transformer (MQTransformer)</p>

## Main Results on ImageNet with Pretrained Models

**DeMT on NYUD-v2 dataset**

| model|backbone|#params| FLOPs | SemSeg| Depth | Noemal|Boundary| model checkpopint | log |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |:---: |
| DeMT |HRNet-18| 4.76M  | 22.07G  | 39.18 | 0.5922 | 20.21| 76.4 | [Google Drive]() | [log]()  |
| DeMT | Swin-T | 32.07M | 100.70G | 46.36 | 0.5871 | 20.60| 76.9 | [Google Drive](https://drive.google.com/file/d/1IfQRVyvaVkEfybzh4QAz9Vq_0U38Hngq/view?usp=share_link) | [log](https://drive.google.com/file/d/1eAtQVJLcvIOMwAfKyl2NmYfe3hPne_WK/view?usp=share_link)  |
| DeMT(xd=2) | Swin-T | 36.6M| - | 47.45 | 0.5563| 19.90| 77.0 | [Google Drive](https://drive.google.com/file/d/1Rz4R9vu8bGtskpJDlVfgexYZoHtz8j8k/view?usp=share_link) | [log](https://drive.google.com/file/d/1TPo4pMjbhPAn3gxKOt4P7hVSPJe1Lpsn/view?usp=share_link)  |
| DeMT | Swin-S | 53.03M | 121.05G | 51.50 | 0.5474 | 20.02 | 78.1 | [Google Drive](https://drive.google.com/drive/folders/1jINF9WOyILqrPcsprWbM5VSCEWozsc1c) | [log](https://drive.google.com/drive/folders/1jINF9WOyILqrPcsprWbM5VSCEWozsc1c)|
| DeMT | Swin-B | 90.9M | 153.65G | 54.34 | 0.5209 | 19.21 | 78.5 | [Google Drive]() | [log]() |
| DeMT | Swin-L | 201.64M | -G | 56.94 | 0.5007 | 19.14 | 78.8 | [Google Drive]() | [log]() |

**DeMT on PASCAL-Contex dataset**

| model | backbone |  SemSeg | PartSeg | Sal | Normal| Boundary| 
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| DeMT |HRNet-18| 59.23 | 57.93 | 83.93| 14.02 | 69.80 |
| DeMT | Swin-T | 69.71 | 57.18 | 82.63| 14.56 | 71.20 |
| DeMT | Swin-S | 72.01 | 58.96 | 83.20| 14.57 | 72.10 | 
| DeMT | Swin-B | 75.33 | 63.11 | 83.42| 14.54 | 73.20 |



## Citing DeMT multi-task method

```
@inproceedings{2022MQTransformer,
  title={Multi-Task Learning with Multi-Query Transformer for Dense Prediction},
  author={Xu, Yangyang and Li, Xiangtai and Yuan, Haobo and Yang, Yibo and Zhang, Lefei },
  journal={arXiv preprint arXiv:2205.14354},
  year={2022}
}
```



**Train**

To train MQTransformer model:
```
python ./src/main.py --cfg ./config/t-nyud/swin/swin_large_mqformer_lr0001.yaml --datamodule.data_dir $DATA_DIR --trainer.gpus 8
```

**Evaluation**

- When the training is finished, the boundary predictions are saved in the following directory: ./logger/NYUD_xxx/version_x/edge_preds/ .
- The evaluation of boundary detection use the MATLAB-based [SEISM](https://github.com/jponttuset/seism) repository to obtain the optimal-dataset-scale-F-measure (odsF) scores.


## Acknowledgement
This repository is based [ATRC](https://github.com/brdav/atrc). Thanks to [ATRC](https://github.com/brdav/atrc)!

