import torch as t

nn = t.nn
f = nn.Functional

class SequentialVAE(nn.Module):
    eps = 1e-20
    def __init__(self, config) -> None:
        self.n_class = config.n_class
        self.n_dist = config.n_dist
        self.tau = config.tau
        
        self.encoder = None 
        self.decoder = None
        self.head = None 
        
    def set_tau(self, value) -> None:
        self.tau = value

    def encode(self):
        pass

    def decode(self):
        pass

    def reparameterize(self, z):
        u = t.rand_like(z)
        g = -t.log(-t.log(u + self.eps) + self.eps)

        logits = f.softmax((z + g) / self.tau, dim=-1)
        return logits.view(-1, self.n_dist * self.n_class)

    def sample(self):
        pass

    def generate(self):
        pass

    def classify(self):
        pass

    def forward(self):
        pass
    
    def loss(self):
        pass
