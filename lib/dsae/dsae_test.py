"""
For N images, where each image can be summarised with F features, calculate the average
distance of each feature point to our expected center crop point for that image. This gives us NxF data points, which we
average to obtain the average distance of each feature, size 1xF. This gives us how
good each feature is. We can choose the minimum of this as the score.

If a feature consistently tracks the object, then its distance to the expected crop should be the smallest across all
demonstrations and trajectories.
"""
import torch


class DSAE_FeatureTest(object):
    def __init__(self, model, size, device):
        self.model = model
        self.h, self.w = size
        # original device model was on, to restore after test
        self.device = device

    def test(self, test_dataloader):
        self.model.eval()
        self.model.cpu()
        with torch.no_grad():
            l2_distances = []
            l1_distances = []
            for batch_idx, batch in enumerate(test_dataloader):
                # (B, 2) in [0-127, 0-96]
                # TODO: batch pixel is a list of a single tensor, this might be due to dataset returning tuple rather than array, check and fix
                pixels = batch["pixel"]
                image_centers = batch["images"][:, 1]
                # (B, C, 2) in [-1, 1] range
                unnormalised_features = self.model.encoder(image_centers)
                normalised = (unnormalised_features + 1) / 2
                features = normalised * torch.tensor([self.w - 1, self.h - 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # (B, C, 2) - (B, 1, 2) -> (B, C, 2)
                differences = (features.type(torch.int32) - pixels.unsqueeze(1)).type(torch.float32)
                # (B, C)
                l2_error_pixels = torch.norm(differences, dim=-1)
                l2_distances.append(l2_error_pixels)
                l1_error_pixels = torch.norm(differences, p=1, dim=-1)
                l1_distances.append(l1_error_pixels)

            # now, cat them and compute average across all images
            # size (N, F), F errors for each of N images
            average_l2 = torch.mean(torch.cat(l2_distances), dim=0)
            average_l1 = torch.mean(torch.cat(l1_distances), dim=0)
        self.model.to(self.device)

        return average_l2, average_l1
