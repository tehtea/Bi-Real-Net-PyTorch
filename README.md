# Bi-Real-Net

This is an alternative implementation of the [Bi-Real-Net paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/zechun_liu_Bi-Real_Net_Enhancing_ECCV_2018_paper.pdf) by Z.C. Liu et al.

It is designed such that the resultant model can be exported into ONNX and then into daBNN for deployment and inference on ARM devices.

# Usage

## Setup
1. Download the ILSVRC2012 datasets here:
- [Train](https://academictorrents.com/details/d58437a61c1adf9801df99c6a82960d076cb7312)
- [Val](https://academictorrents.com/details/207ebd69f80a3707f035cd91a114466a270e044d)

2. Install the required dependencies
```sh
pip install -r requirements.txt
```

## For training
```sh
python3 main_train.py --train_path <Path to train .lmdb file> --val_path <Path to val .lmdb file>
```
Other command line arguments can be found by running `python3 main_train.py --help`.

# Environment
- CUDA 9.0 (change this by changing the url in requirements.txt)
- Python 3.5.2
- PyTorch 1.0.0 (needed to get ONNX model with IR Version 3 and Opset Version 9, needed by ONNX2daBNN converter)

# References
- Loading LMDB dataset: https://github.com/rmccorm4/PyTorch-LMDB
- Original Bi-Real-Net implementation: https://github.com/liuzechun/Bi-Real-net
- Binary Network training in PyTorch: https://github.com/jiecaoyu/XNOR-Net-PyTorch
- daBNN Conv2d implementation in PyTorch: https://gist.github.com/daquexian/7db1e7f1e0a92ab13ac1ad028233a9eb
- Network model visualization and debugging: https://netron.app/