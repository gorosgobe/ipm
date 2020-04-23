import torch


class FeatureProvider(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def __call__(self, x):
        # make sure weights are frozen
        self.model.eval()
        with torch.no_grad():
            if len(x.size()) == 3:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            # returns (B, C*2 = latent dimension)
            return self.model.encoder(x)
