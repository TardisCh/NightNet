# NightNet

`NightNet` is a deep cross-sensor reconstruction network designed to generate VIIRS-like nighttime light data from historical DMSP observations with Landsat and impervious surface auxiliary inputs.

This repository provides the official code for the NightNet model presented in the manuscript below.

NightNet was developed for the reconstruction of long-term monthly nighttime light products described in the manuscript:

*A Temporally Consistent Global 500 m-Resolution Monthly VIIRS-Like Nighttime Light Dataset (1992--2024)*.

## Included files

- `NightNets.py`: main model definition
- `base/cbam.py`: CBAM attention block used in the decoder

## Network overview

NightNet is a multi-input image reconstruction network built around an encoder-decoder structure.

Within the MVNL framework described in the paper, NightNet is used to reconstruct VIIRS-like nighttime light products from historical DMSP observations. The goal is to bridge the gap between the long temporal coverage of DMSP-OLS and the finer spatial and radiometric quality of NPP-VIIRS.

The network is designed for cross-sensor nighttime light reconstruction with multi-modal auxiliary observations. In the released code, it produces a main reconstruction output together with two auxiliary prediction branches.

Based on the current implementation, the model takes three inputs:

- low-resolution nighttime light data
- multi-band Landsat features
- impervious surface information

## Input and output

Model forward interface:

```python
output, edge_out, mask_out = model(x_LR, x_LSAT, x_IS)
```

Expected inputs:

- `x_LR`: low-resolution DMSP nighttime light input
- `x_LSAT`: multi-band Landsat auxiliary input
- `x_IS`: impervious surface auxiliary input

Returned outputs:

- `output`: reconstructed VIIRS-like data
- `edge_out`: edge branch output
- `mask_out`: mask branch output

## Minimal usage

```python
from models.nets.NightNets import NightNets

model = NightNets(in_channel=1, out_channel=1)
```

## Dependency notes

This module relies mainly on:

- `torch`
- `torchvision.ops.DeformConv2d`

## Citation

If you use NightNet in your research, please cite it as follows:

```bibtex
@article{cheng2026mvnl,
  author = {Cheng, H. and Geng, M. and Li, X. and Li, S. and Zhao, M. and Lin, C. and Wang, J. and Gong, P. and Zhou, Y.},
  title = {A Temporally Consistent Global 500 m-Resolution Monthly VIIRS-Like Nighttime Light Dataset (1992--2024)},
  journal = {Earth System Science Data Discussions},
  year = {2026},
  doi = {10.5194/essd-2026-129},
  note = {Preprint, in review}
}
```
