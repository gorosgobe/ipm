import torch


class FeatureProvider(object):
    def __init__(self, model, device):
        self.model = model
        # if device is None, input tensor is not moved
        self.device = device

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def cpu(self):
        self.model.cpu()

    def to(self, device):
        self.model.to(device)

    def __call__(self, x):
        # make sure weights are frozen
        self.model.eval()
        with torch.no_grad():
            if len(x.size()) == 3:
                x = x.unsqueeze(0)

            if self.device is not None:
                x = x.to(self.device)
            # returns (B, C*2 = latent dimension)
            return self.model.encoder(x)


class RLFeatureProvider(FeatureProvider):
    def __init__(self, feature_provider_model, device, rl_model):
        super().__init__(feature_provider_model, device)
        self.rl_model = rl_model

    def __call__(self, x):
        features = super().__call__(x)
        # TODO: use features and rl model to return filtered features
        return None

