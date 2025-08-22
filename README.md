<div align="center">

# 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection

<a href="https://arxiv.org/abs/2507.23567"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://royyang0714.github.io/3D-MOOD'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>

</div>

<div>
  <img src="assets/overview.png" width="100%" alt="Banner 2" align="center">
</div>

<div>
  <p></p>
</div>

> [**3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection**](https://royyang0714.github.io/3D-MOOD) \
> Yung-Hsu Yang, Luigi Piccinelli, Mattia Segu, Siyuan Li, Rui Huang, Yuqian Fu, Marc Pollefeys, Hermann Blum, Zuria Bauer \
> ICCV 2025,
> *Paper at [arXiv 2507.23567](https://arxiv.org/pdf/2507.23567.pdf)*


## News and ToDo

- [ ] Release code and models.
- [x] `25.06.2025`: 3D-MOOD is accepted at ICCV 2025!

## Getting Started

### Installation

We support Python 3.11+ and PyTorch 2.4.0+. Please install the correct PyTorch version according to your own hardware settings.

```bash
conda create -n opendet3d python=3.11 -y

conda activate opendet3d

# Install Vis4D
# It should also install the PyTorch with CUDA support. But please check.
pip install vis4d

# Install CUDA ops
pip install git+https://github.com/SysCV/vis4d_cuda_ops.git

# Install 3D-MOOD
pip install -v -e .
```

### Data Preparation

## Citation

If you find our work useful in your research please consider citing our publications:
```bibtex
@article{yang20253d,
  title={3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection},
  author={Yang, Yung-Hsu and Piccinelli, Luigi and Segu, Mattia and Li, Siyuan and Huang, Rui and Fu, Yuqian and Pollefeys, Marc and Blum, Hermann and Bauer, Zuria},
  journal={arXiv preprint arXiv:2507.23567},
  year={2025}
}
```
