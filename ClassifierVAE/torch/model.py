import torch as t
import numpy as np

nn = t.nn
f = nn.Functional

class SequentialVAE(nn.Module):
    eps = 1e-20
    def __init__(self, config) -> None:
        self.n_class = config.n_class
        self.n_dist = config.n_dist
        self.tau = config.tau
        
        self.encoder = config.encoder()
        self.decoder = config.decoder()
        self.head = config.head()

        self.fc_z = nn.Linear(config.latent*4, self.n_dist * self.n_class)
        self.scale = nn.Linear(self.n_dist * self.n_class, config.latent*4)

        
    def set_tau(self, value) -> None:
        self.tau = value

    def encode(self, input):
        latent = self.encoder(input)
        latent = t.flatten(latent, start_dim=1)

        z = self.fc_z(latent)
        z = z.view(-1, self.n_dist, self.n_class)

        return [z]

    def decode(self, z):
        x = self.scale(z)
        x = x.view(-1, self.latent, 2, 2)

        decoded = self.decoder(x)

        return decoded

    def reparameterize(self, z):
        u = t.rand_like(z)
        g = -t.log(-t.log(u + self.eps) + self.eps)

        logits = f.softmax((z + g) / self.tau, dim=-1)
        return logits.view(-1, self.n_dist * self.n_class)

    def sample(self):
        return None

    def generate(self):
        return None

    def forward(self, input):
        q = self.encode(input)[0]
        z = self.reparameterize(q)
        recons = self.decode(z)
        y_pred = self.predict(recons)
        return [recons, input, q, y_pred]
    
    def predict(self, input):
        return self.head(input)
    
    def loss(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        q = args[2]
        y_pred = args[3]

        q_p = f.softmax(q, dim=-1)

        recons_loss = f.mse_loss(recons, input, reduction='mean')

        h1 = q_p * t.log(q_p + self.eps)
        h2 = q_p * np.log(1. / self.n_dist + self.eps)

        kl = q_p * np.log(1. / self.categorical_dim + self.eps)

        loss = self.alpha * recons_loss + kld_weight * kl

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':-kl}

