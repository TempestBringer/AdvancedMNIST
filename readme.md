## 数据集 
1. [Extended MNIST]https://www.kaggle.com/datasets/crawford/emnist?resource=download
2. [HASYv2]https://zenodo.org/records/259444#.Xe4DVZMzbOQ
3. [Handwritten math symbols dataset]https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols

## 训练

```commandline
python train.py --batch_size 4 --device cuda:0 --save_ckpt_folder ./ckpt --save_ckpt_name test.ckpt
```