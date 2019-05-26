import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vgg

from collections import namedtuple

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=4),
            nn.ReLU(inplace=True),
            
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            
            ConvolutionalBlock(64, 64, kernel_size=3, stride=1),
            ConvolutionalBlock(64, 64, kernel_size=3, stride=1),
            ConvolutionalBlock(64, 64, kernel_size=9, stride=4),
        )
        
    def forward(self, X):
        return self.model(X)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Con2d(3, 48, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            
            ConvolutionalBlock(48, 128, kernel_size=5, stride=2),
            ConvolutionalBlock(48, 192, kernel_size=3, stride=1),
            ConvolutionalBlock(192, 192, kernel_size=3, stride=1),
            ConvolutionalBlock(192, 128, kernel_size=3, stride=2),
            
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, X):
        return self.model(X)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super(ResidualBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 

            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                      stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return X + self.model(X)


class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvolutionalBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, X):
        return self.model(X)


class WESPE(object):
    def __init__(self, opt):
        self.opt = opt
        self.set_device(opt)

        self.G = Generator()
        self.F = Generator()

        self.D_c = Discriminator()
        self.D_t = Discriminator()

    def set_device(self, opt):
        self.device = torch.cuda.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.cuda.device('cpu')

    def set_input(self, input):
        self.real_X = input['X'].to(self.device)
        self.real_Y = input['Y'].to(self.device)

    def kernel_to_conv2d(self, kernel_size, nsig, channels):
        out_filter = gaussian_kernel(kernel_size, nsig, channels)
        gaussian_filter = nn.Conv2d(channels, channels, kernel_size, groups=channels, bias=False, padding=10)
        gaussian_filter.weight.data = out_filter
        gaussian_filter.weight.requires_grad = False
        return gaussian_filter
        
        
    def blur(self, img, kernel_size, sigma, channels, device):
        out_filter = self.kernel_to_conv2d(kernel_size, sigma, channels).to(device)
        return out_filter(img)
    
    def grayscale(self, img):
        return torch.unsqueeze(img[:, 0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114, 1)
        
    def forward(self, input):
        self.set_input(input)

        bce_loss = nn.BCELoss()
        mse_loss = nn.MSELoss()

        self.enhanced_Y = self.G(self.real_X)
        self.reconstructed_X = self.F(self.enhanced_Y)

        content_loss_net = LossNetwork()
        self.c_reconstrucion_loss = content_loss_net(self.reconstructed_X)
        self.c_real_loss = content_loss_net(self.real_X)
        
        self.G_loss = mse_loss(self.c_reconstrucion_loss, self.c_real_loss)

        self.blur_enhanced_Y = self.blur(self.enhanced_Y, self.opt.kernel_size, self.opt.sigma, self.opt.channels, self.device)
        self.D_c_logit_enhanced_Y = self.D_c(self.blur_enhanced_Y)
        self.D_c_enhanced = bce_loss(self.D_c_logit_enhanced_Y, 0)

        self.blur_real_Y = self.blur(self.real_Y)
        self.D_c_logit_real_Y = self.D_c(self.blur_real_Y)
        self.D_c_real = bce_loss(self.D_c_logit_real_Y, 1)

        self.D_c_loss = self.D_c_enhanced + self.D_c_real

        self.gray_enhanced_Y = self.grayscale(self.enhanced_Y)
        self.D_t_logit_enhanced_Y = self.D_t(self.gray_enhanced_Y)
        self.D_t_enhanced = bce_loss(self.D_t_logit_enhanced_Y, 0)

        self.gray_real_Y = self.grayscale(self.real_Y)
        self.D_t_logit_real_Y = self.D_t(self.gray_real_Y)
        self.D_t_real = bce_loss(self.D_t_logit_real_Y, 1)

        self.D_t_loss = self.D_t_enhanced + self.D_t_real




LossOutput = namedtuple(
    "LossOutput", ["relu1", "relu2", "relu3", "relu4", "relu5"])


class LossNetwork(torch.nn.Module):
    def __init__(self):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg.vgg19(pretrained=True).features
        self.layer_name_mapping = {
            '3': "relu1",
            '8': "relu2",
            '17': "relu3",
            '26': "relu4",
            '35': "relu5",
        }

    def forward(self, X):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            X = module(X)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = X
        return LossOutput(**output)