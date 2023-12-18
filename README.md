# $D^2$ EFS Deep Depth Estimation For Stereo Cameras

Deep Learning model for stereo depth estimation. The model employs UNet architecture for fine depth estimation. The model allows you to either use the transpose convolution to reconstruct the depth image or use bilinear interpolation with same-padded CNNs. The pretrained model uses the version with the transpose convolution as it was the one yieled the highest accuracy.

## How to use 
1. Run `python -m venv venv` in the project directory
2.  On Linux use `source venv/bin/activate`. on Windows, use `venv\Scripts\activate`
To use the notebooks, make sure that you have anaconda installed and only run the first command in the previous list after being in the base environment.

## Model Parameters
Download the model pretrained parameters from [here](https://website-name.com)
