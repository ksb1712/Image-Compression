# Image-Compression
## Simple Image compression using CNN in Pytorch.
Tested in Python 3.6
<br>
Requirements: Pytorch, skimage, PIL, patchify, opencv
## Theory
General image compression programs using deep learning,to try and reduce the image dimensionality by learning the latent space representations. Here a thumbnail image of the original image is first created by downscaling the image and then the residual between the thumbnail and the orignal image is encoded in the latent space. This results in better encoding via an entropy encoder as most of the values are 0. 

## Usage

```
import test_utils as utils
utils.compress_img(name.png) 
utils.decompress_img(name.npz)
```


