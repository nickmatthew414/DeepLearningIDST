import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
                                   nn.InstanceNorm2d(256), nn.ReLU(),
                                   nn.ReflectionPad2d(1),
                                   nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
                                   nn.InstanceNorm2d(256))

    def forward(self, x):
        return x + self.block(x)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, stride=1)
        )

    def forward(self, x):
        return self.decoder(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = Encoder()
        blocks = [ResidualBlock() for i in range(6)]
        self.residual_block = nn.Sequential(*blocks)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.residual_block(x)
        return self.decoder(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2),
            # nn.InstanceNorm2d(64), # paper says it doesn't use instance norm in the first block
            nn.LeakyReLU(.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(.2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(.2),
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)