# Some utils for data loading etc

#    Copyright (C) 2024 MI2G
#    Klatzer, Teresa t.klatzer@sms.ed.ac.uk
#    Melidonis, Savvas sm2041@hw.ac.uk
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
from natsort import os_sorted
import torch
import hdf5storage
import numpy as np
import torch.nn.functional as F


# following code is modified from
# https://github.com/samuro95/Prox-PnP
# and 
# https://github.com/samuro95/GSPnP
# and
# https://github.com/samuro95/BregmanPnP
#
# Papers: [1] "Proximal denoiser for convergent plug-and-play optimization with nonconvex regularization" published at ICML 2022.
# by Samuel Hurault, Arthur Leclaire, Nicolas Papadakis.
# and
# [2] "Gradient Step Denoiser for convergent Plug-and-Play" published at ICLR 2022.
# Samuel Hurault, Arthur Leclaire, Nicolas Papadakis.
# [3] "Convergent Bregman Plug-and-Play Image Restoration for Poisson Inverse Problems" presented at Neurips 2023.
# Samuel Hurault, Arthur Leclaire, Nicolas Papadakis.

def prepare_data_paths(config):
    input_path = os.path.join(config.dataset_path, config.dataset_name, "0")
    return os_sorted(
        [os.path.join(input_path, p) for p in os.listdir(input_path)]
    )


def load_kernels(config):

    k_list = []
    # load the 8 motion blur kernels
    kernel_path = os.path.join(config.kernel_path, config.kernel_name)
    kernels = hdf5storage.loadmat(kernel_path)["kernels"]
    # Kernels follow the order given in the paper [2] (Table 2). 
    # The 8 first kernels are motion blur kernels, the 9th kernel is uniform and the 10th Gaussian.
    # The 11th kernel is a dummy kernel to run denoising.
    for k_index in range(11):
        if k_index == 8:  # Uniform blur
            k = np.float32((1 / 81) * np.ones((9, 9)))
        elif k_index == 9:  # Gaussian blur
            k = np.float32(matlab_style_gauss2D(shape=(25, 25), sigma=1.6))
        elif k_index == 10:
            k = np.zeros((3,3)).astype(np.float32) # dummy kernel for denoising
            k[1,1] = 1
        else:  # Motion blur
            k = np.float32(kernels[0, k_index])
        k_list.append(k)

    k_list = [torch.from_numpy(item).float().to(config.device) for item in k_list]
    return k_list


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def At(y, k_tensor):
    """
    Calculation A*x with A the linear degradation operator 
    """
    return Gt(y, k_tensor, sf=1)

def A(y, k_tensor):
    """
    Calculation A*x with A the linear degradation operator 
    """
    return G(y, k_tensor, sf=1)


def G(x, k, sf=3):
    """
    x: image, NxcxHxW
    k: kernel, hxw
    sf: scale factor
    center: the first one or the moddle one
    Matlab function:
    tmp = imfilter(x,h,'circular');
    y = downsample2(tmp,K);
    """
    x = downsample(imfilter(x, k), sf=sf)
    return x


def Gt(x, k, sf=3):
    """
    x: image, NxcxHxW
    k: kernel, hxw
    sf: scale factor
    center: the first one or the moddle one
    Matlab function:
    tmp = upsample2(x,K);
    y = imfilter(tmp,h,'circular');
    """
    x = imfilter(upsample(x, sf=sf), k, transposed=True)
    return x

def downsample(x, sf=3):
    """s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    """
    st = 0
    return x[..., st::sf, st::sf]

def upsample(x, sf=3):
    """s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    """
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * sf, x.shape[3] * sf)).type_as(
        x
    )
    z[..., st::sf, st::sf].copy_(x)
    return z

def imfilter(x, k, transposed=False, n_channel=3):
    """
    Equivalent (verified) to scipy ndimage.convolve with mode='wrap'.
    x: image, NxcxHxW
    k: kernel, hxw
    """
    n_channel = x.shape[1]
    k = k.repeat(n_channel, 1, 1, 1)
    k = k.flip(-1).flip(-2)  # flip kernel for convolution and not correlation !!!
    ph = (k.shape[-2] - 1) // 2
    pw = (k.shape[-1] - 1) // 2
    if not transposed:
        x = pad_circular(x, padding=(ph, pw))
        x = F.conv2d(x, k, groups=x.shape[1])
    else:
        x = F.conv_transpose2d(x, k, groups=x.shape[1])
        x = unpad_circular(x, padding=(ph, pw))
    return x

def pad_circular(input, padding):
    # type: (Tensor, List[int]) -> Tensor
    """
    Arguments
    :param input: tensor of shape :math:`(N, C_{\text{in}}, H, [W, D]))`
    :param padding: (tuple): m-elem tuple where m is the degree of convolution
    Returns
    :return: tensor of shape :math:`(N, C_{\text{in}}, [D + 2 * padding[0],
                                     H + 2 * padding[1]], W + 2 * padding[2]))`
    """
    offset = 3
    for dimension in range(input.dim() - offset + 1):
        input = dim_pad_circular(input, padding[dimension], dimension + offset)
    return input

def unpad_circular(input, padding):
    ph, pw = padding
    out = input[:, :, ph:-ph, pw:-pw]
    # sides
    out[:, :, :ph, :] += input[:, :, -ph:, pw:-pw]
    out[:, :, -ph:, :] += input[:, :, :ph, pw:-pw]
    out[:, :, :, :pw] += input[:, :, ph:-ph, -pw:]
    out[:, :, :, -pw:] += input[:, :, ph:-ph, :pw]
    # corners
    out[:, :, :ph, :pw] += input[:, :, -ph:, -pw:]
    out[:, :, -ph:, -pw:] += input[:, :, :ph, :pw]
    out[:, :, :ph, -pw:] += input[:, :, -ph:, :pw]
    out[:, :, -ph:, :pw] += input[:, :, :ph, -pw:]
    return out

def dim_pad_circular(input, padding, dimension):
    # type: (Tensor, int, int) -> Tensor
    input = torch.cat(
        [input, input[[slice(None)] * (dimension - 1) + [slice(0, padding)]]],
        dim=dimension - 1,
    )
    input = torch.cat(
        [
            input[[slice(None)] * (dimension - 1) + [slice(-2 * padding, -padding)]],
            input,
        ],
        dim=dimension - 1,
    )
    return input

