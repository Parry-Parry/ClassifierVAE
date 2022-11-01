import torch as t

nn = t.nn

def init_encoder(config):
    def encoder():
        layers = [] 
        in_dim = config.in_dim

        for size in config.stack[:-1]:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, out_channels=size,
                                kernel_size=config.kernel, stride=config.stride, padding=config.padding),
                        nn.BatchNorm2d(size),
                        nn.LeakyReLU()
                )
            )
            in_dim = size
            
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_dim, out_channels=config.stack[-1],
                                kernel_size=config.kernel, stride=config.stride, padding=config.padding)
            )
        )

        return nn.Sequential(*layers)
    return encoder

def init_decoder(config):
    def decoder():
        layers = [] 

        for i in range(len(config.stack) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(config.stack[i], out_channels=config.stack[i+1],
                                kernel_size=config.kernel, stride=config.stride, padding=config.padding, output_padding=1),
                        nn.BatchNorm2d(config.stack[i+1]),
                        nn.LeakyReLU()
                )
            )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(config.stack[-2], out_channels=config.stack[-1],
                                kernel_size=config.kernel, stride=config.stride, padding=config.padding, output_padding=1)
            )
        )

        layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(config.stack[-1],
                                   config.stack[-1],
                                   kernel_size=config.kernel,
                                   stride=config.stride,
                                   padding=config.padding,
                                   output_padding=1),
                nn.BatchNorm2d(config.stack[-1]),
                nn.LeakyReLU(),
                nn.Conv2d(config.stack[-1], out_channels=3,
                            kernel_size=config.kernel, padding=config.padding),
                nn.Tanh())
            )

        return nn.Sequential(*layers)
    return decoder
    
def head():
    pass