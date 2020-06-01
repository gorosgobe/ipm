import torch

from lib.rl.state import FilterSpatialFeatureState


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
        return self

    def to(self, device):
        self.model.to(device)
        return self

    def get_num_parameters(self):
        return sum([p.numel() for p in self.model.encoder.parameters()])

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


class FilterSpatialRLFeatureProvider(FeatureProvider):
    def __init__(self, feature_provider_model, device, rl_model, k):
        super().__init__(feature_provider_model, device)
        self.rl_model = rl_model
        self.k = k

    def __call__(self, x):
        # features are (1, C, 2), torch tensor, need as numpy vector
        features = super().__call__(x).squeeze(0).view(-1)
        # (C*2,)
        np_features = features.cpu().numpy()
        action, _states = self.rl_model.predict(np_features, deterministic=True)
        output_features = FilterSpatialFeatureState(
            k=self.k, spatial_features=np_features
        ).get_top_k_features(action)
        return torch.tensor(output_features).view(1, -1, 2)

