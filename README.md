# Image Restoration Hub
A Image restoration hub based on basicsr library, including: image super resolution, image denoise, etc. All the settings are adjusted on our un-public infrared dataset.

## News
- 2024-10-14. Add inference code to quickly verify model result.

## Supported models
- ECBSR. Real-time super resolution.
- SwinIR. A swin-tansformer based strong image resotration baseline.

## Results
### Subjective effect. 

## Enviroments
- python=3.10.14
- torch=2.4.0
- torchvision=0.19.0

## Installation
```
pip install -r requirements.txt
python setup.py develop
```