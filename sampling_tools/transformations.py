# Some image transformations
# Intended to use in around application of data-driven priors

#    Copyright (C) 2024 MI2G
#    Dobson, Paul pdobson@ed.ac.uk
#    Kemajou, Mbakam Charlesquin cmk2000@hw.ac.uk
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


import random
import torch

def transform_flip_rot(image):
    flip_lr = random.choice([True, False]) 
    if flip_lr: # flip left right
        image = torch.flip(image, dims = (-1,))

    flip_ud = random.choice([True, False])
    if flip_ud: # flip up down
        image = torch.flip(image, dims = (-2,))

    select_rot = random.choice([0, 1, 2, 3])
    rot = (select_rot > 0)
    if rot: # rotate
        image = torch.rot90(image, k=select_rot, dims=(-1,-2))

    return image, flip_lr, flip_ud, rot, select_rot
    
def undo_transform_rot_flip(image, flip_lr, flip_ud, rot, select_rot):

    if rot:
        image = torch.rot90(image, k=-select_rot, dims=(-1,-2))
    if flip_ud:
        image = torch.flip(image, dims = (-2,))
    if flip_lr:
        image = torch.flip(image, dims = (-1,))

    return image
