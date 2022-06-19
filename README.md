# Learning from ambiguous labels for lung nodule malignancy prediction

This repo contains the official implementation of our paper: Learning from ambiguous labels for lung nodule malignancy prediction, which proposes a multi-view 'divide-and-rule' (MV-DAR) model to learn from both reliable and ambiguous annotations for lung nodule malignancy prediction on chest CT scans. The implementation of DAR model is released.
<p align="center"><img src="https://raw.githubusercontent.com/Merrical/DAR/master/MVDAR_overview.png" width="90%"></p>

#### [Paper on IEEE](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9705525)
#### [Paper on arXiv](https://arxiv.org/pdf/2104.11436.pdf)

### Requirements
This repo was tested with Ubuntu 20.04.4 LTS, Python 3.8, PyTorch 1.9.0, and CUDA 10.1.
We suggest using virtual env to configure the experimental environment.

1. Clone this repo:

```bash
git clone https://github.com/Merrical/DAR_code.git
```

2. Create experimental environment using virtual env:

```bash
virtualenv .env --python=3.8 # create
source .env/bin/activate # activate
pip install -r requirements.txt
```

### Bibtex
```
@article{liao2022learning,
  title={Learning from ambiguous labels for lung nodule malignancy prediction},
  author={Liao, Zehui and Xie, Yutong and Hu, Shishuai and Xia, Yong},
  journal={IEEE Transactions on Medical Imaging},
  year={2022},
  publisher={IEEE}
}
```

### Contact Us
If you have any questions, please contact us ( merrical@mail.nwpu.edu.cn ).