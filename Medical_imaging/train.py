"""
Trains a 3D UNet model on a medical image segmentation task using the MONAI library.

Example:
    ```
    python train_unet.py
    ```

"""
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from preporcess import prepare
from Medical_imaging.utilities import train
from dotenv import load_dotenv
import os

load_dotenv()

data_dir = os.getenv('DATA_DIR')
model_dir = os.getenv('MODEL_DIR')
data_in = prepare(data_dir, cache=True)

device = torch.device("cuda:0")

"""
Defines the UNet model architecture.

Args:
    dimensions (int): Number of dimensions in the input data (3 for 3D medical images)
    in_channels (int): Number of input channels (1 for grayscale images)
    out_channels (int): Number of output channels (2 for binary segmentation)
    channels (tuple): Number of channels in each layer of the UNet
    strides (tuple): Strides for downsampling in each layer of the UNet
    num_res_units (int): Number of residual units in each layer of the UNet
    norm (Norm): Normalization layer to use (BatchNorm in this case)

Returns:
    UNet: The defined UNet model
"""
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)


optimizer = torch.optim.Adam(model.parameters(), 1e-5, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':

    train(model, data_in, loss_function, optimizer, 600, model_dir)