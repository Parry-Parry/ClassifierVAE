import torch as t

nn = t.nn

class SequentialVAE(nn.Module):
    eps = 1e-20
    def __init__(self) -> None:
        self.n_class = None 
        self.n_dist = None 
        
        self.encoder = None 
        self.decoder = None
        self.head = None 
        

    def encode(self):
        pass
    def decode(self):
        pass
    def reparameterize(self, z):
        u = t.rand_like(z)
        g = -t.log
    def sample(self):
        pass
    def generate(self):
        pass
    def classify(self):
        pass
    def forward(self):
        pass