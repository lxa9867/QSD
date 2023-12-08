<p align="center"><img src="teaser.png" width="700"/></p>

> [**Towards Robust Audiovisual Segmentation in Complex Environments with Quantization-based Semantic Decomposition**](https://arxiv.org/pdf/2310.00132.pdf)
>
> Xiang Li, Jinglu Wang, Xiaohao Xu, Xiulian Peng, Rita Singh, Yan Lu, Bhiksha Raj
---

## Updates
- **(2023-12-07)** Repo created. We will release the code soon.

## Dataset
Download the AVS and AVSS datasets from [AVSBench](http://www.avlbench.opennlplab.cn/leaderboard/avss).

## Install
```
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 -c pytorch
pip install -r requirements.txt 
pip install 'git+https://github.com/facebookresearch/fvcore' 
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
cd models/ops
python setup.py build install
cd ../..
```

## Docker
You may try [docker](https://hub.docker.com/r/ang9867/refer) for a quick start.

## Audiovisual Semantic Segmentation (AVSS)
```
bash ./scripts/dist_train_avss_local.sh $out_path$ $weight_path$/r50_pretrained.pth --backbone resnet50 --as_avs True --quantitize_query True --fpn_type 'audio_dual' --global_decompose_query True -quantitize_query True --fpn_type 'audio_dual' --global_decompose_query True --dataset_file 'avss'
```

## Audiovisual Segmentation (AVS)
```
bash ./scripts/dist_train_avs_local.sh $out_path$ $weight_path$/r50_pretrained.pth --backbone resnet50 --as_avs True --global_decompose_query True --quantitize_query True --fpn_type 'audio_dual' --binary --dataset_file 'avs_1s7m'
```

## Citation

```
@article{li2023rethinking,
  title={Rethinking Audiovisual Segmentation with Semantic Quantization and Decomposition},
  author={Li, Xiang and Wang, Jinglu and Xu, Xiaohao and Peng, Xiulian and Singh, Rita and Lu, Yan and Raj, Bhiksha},
  journal={arXiv preprint arXiv:2310.00132},
  year={2023}
}
```
