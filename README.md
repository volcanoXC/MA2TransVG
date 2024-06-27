# Multi-Attribute Interactions Matter for 3D Visual Grounding

## Installation
1. The code is now compatible with PyTorch 1.10. You can follow the [instructions](https://cshizhe.github.io/projects/vil3dref.html) to build the environment.
```
conda create -n MA2Trans python=3.8
conda activate MA2Trans

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```
2. To use a PointNet++ visual encoder, you need to compile its CUDA layers for PointNet++.
```
cd lib/pointnet2
python setup.py install
```
3. We adopt bert-base-uncased from huggingface, which can be installed using pip as follows:
```
pip install transformers
```

## Data Preparation
1. For the ScanRefer dataset, you can access the original ScanNet dataset and please refer to the [ScanNet Instructions](https://forms.gle/aLtzXN12DsYDMSXX6). The data format is as:
```
"scene_id": [ScanNet scene id, e.g. "scene0000_00"],
"object_id": [ScanNet object id (corresponds to "objectId" in ScanNet aggregation file), e.g. "34"],
"object_name": [ScanNet object name (corresponds to "label" in ScanNet aggregation file), e.g. "coffee_table"],
"ann_id": [description id, e.g. "1"],
"description": [...],
"token": [a list of tokens from the tokenized description]
```
2. For Nr3D and Sr3D datasets, you can refer the data preparation from [referit3d](https://github.com/referit3d/referit3d).
3. Please follow the data preprocess in [vil3dref](https://cshizhe.github.io/projects/vil3dref.html) and change the PROCESSED_DATA_DIR folder according to your setting.
4. You can download the pre-trained weight in [this page](https://huggingface.co/bert-base-uncased/tree/main) and put them into the PATH_OF_BERT folder according to your setting.

## Training
* Change the DATA_DIR for different datasets and do the following command:
```
python main.py
```

## Evaluation
* The program will automatically evaluate the performance of the current model and save the best model.

## Citation
If you find this work useful, please consider citing:
```
@InProceedings{xu2024multi,
author       = {Xu, Can and Han, Yuehui and Xu, Rui and Hui, Le and Xie, Jin and Yang, Jian},
title        = {Multi Attributes Interactions Matters for 3D Visual Grounding},
booktitle    = {CVPR},
year         = {2024},
}
```
## Acknowledgement
Some of the codes are built upon [referit3d](https://github.com/referit3d/referit3d) and thanks for the great work.
