## Adversarial Binary Coding for Efficient ReID
This project contains the codes for the paper [Adversarial Binary Coding for Efficient Person Re-identification](http://arxiv.org/abs/1803.10914). We adopt adversarial learning to obtain compact discriminative binary representation for pedestrians and use them to measure similarity in an unified deep learning framework. The method is tested on CUHK03, Market-1501, and DukeMTMC-reID datasets.

## Requirements
The codes are tested in the Anaconda 4.4.0 environment containing following packages:
- Python 3.6
- PyTorch 0.3.0 + torchvision 0.12

## Usage
- Train and test.
```Shell
python main.py --dataset market1501 --train --test --trial 1 --data_dir /root/to/data
```
- ```draw.py``` for drawing loss curves.

By default, the '--data_dir' directs the 'data' folder in the root of the codes. Users can put the data folders into the 'data' folder. For convenience, we have put the train/test splitting file in pickle-readable format into the cuhk03 data folder.

## Exemplar Loss Curves on Market-1501   
![](https://github.com/dongb5/AdversarialBinaryCoding4ReID/blob/master/figs/triplet_loss_small.jpg?raw=true)  ![](https://github.com/dongb5/AdversarialBinaryCoding4ReID/blob/master/figs/D_loss_small.jpg?raw=true) ![](https://github.com/dongb5/AdversarialBinaryCoding4ReID/blob/master/figs/G_loss_small.jpg?raw=true)

## Citation
@article{liu2018abc,   
  title={Adversarial Binary Coding for Efficient Person Re-identification},   
  author={Liu, Zheng and Qin, Jie and Li, Annan and Wang, Yunhong and Van Gool, Luc},   
  journal={arXiv preprint arXiv:1803.10914},   
  year={2018}   
}
