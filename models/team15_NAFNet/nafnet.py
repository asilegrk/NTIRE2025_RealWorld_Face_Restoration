# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import glob
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFNet(nn.Module):

    def __init__(self, img_channel=3, out_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp
        x = x[:, :, :H, :W]
        return {"img": x}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def nafnet(cfg):
    print ("Creating Nafnet")
    num_channels= cfg["in_ch"]
    out_channels= cfg["out_ch"]
    width   = cfg["width"]
    enc_blks = cfg["enc_blks"]
    middle_blk_num = cfg["middle_blk_num"]
    dec_blks = cfg["dec_blks"]
    
    model = NAFNet(img_channel=num_channels, out_channel=out_channels, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)

    return model



########### NAFNET + LEARNED POST-PROC #########

class NAFnetPP(nn.Module):
    """
    Wrapper to train post-processing / refinement models using a base model (in this example NAFNet).
    In this case, the refinement model should add more details to the images without the raindrops.
    The base model is not trained, it is typically frozen. 
    """
    def __init__(self, cfg):
        super(NAFnetPP, self).__init__()

        num_channels= cfg["in_ch"]
        out_channels= cfg["out_ch"]
        width       = cfg["width"]
        enc_blks    = cfg["enc_blks"]
        middle_blk_num = cfg["middle_blk_num"]
        dec_blks    = cfg["dec_blks"]
        train_base  = cfg["base_train"]
    
        self.base = NAFNet(img_channel=num_channels, out_channel=out_channels, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
        
        # very important to start using the pre-trained base model
        self.base.load_state_dict(torch.load(cfg["base_weights"], map_location="cpu"), strict=True)
        
        # In most cases, the base model is not trained, it is frozen
        if not train_base:
            for param in self.base.parameters():
                param.requires_grad = False

        ######## Post-processing / Refinement blocks or NN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        
    def forward(self, x):
        base_result = self.base(x)["img"]
        y = self.conv1(base_result)
        y = self.conv2(y)
        y = self.conv3(y)
        out = y + base_result
        return {"img":out}





def main(model_dir, input_path, output_path, device):
    """
    Face restoration model function for NTIRE2025 competition
    Args:
        model_dir: Path to pretrained model weights
        input_path: Input folder containing PNG images
        output_path: Output folder to save restored images
        device: Computation device (cuda/cpu)
    """

    # --- Model Initialization ---
    model = NAFNet(img_channel=3, out_channel=3, width=32, middle_blk_num=1,
                      enc_blk_nums=[2,2,2], dec_blk_nums=[1,1,1])

    # --- Load Weights ---
    weights_path = os.path.join(model_dir, "nafacev2.pt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    # --- Preprocessing ---
    preprocess = transforms.Compose([
        transforms.ToTensor(),  
    ])

    # --- Fix Output Path ---
    output_path = os.path.dirname(output_path)  
    final_output_path = os.path.join(output_path, os.path.basename(input_path))  
    os.makedirs(final_output_path, exist_ok=True) 

    # --- Process all PNG images ---
    for img_path in glob.glob(os.path.join(input_path, "*.png")):
        img_name = os.path.basename(img_path)

        # Load and preprocess
        input_img = Image.open(img_path).convert("RGB")
        input_tensor = preprocess(input_img).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            output_dict = model(input_tensor)
            output_tensor = output_dict["img"] 

        # Save tensor output directly without transformation
        save_image(output_tensor, os.path.join(final_output_path, img_name), normalize=False)
