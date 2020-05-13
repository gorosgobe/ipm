import argparse
import os

from torchvision.transforms import transforms

from lib.bop.crop_size_feature_search import CropSizeFeatureSearch
from lib.common.utils import get_seed, set_up_cuda
from lib.dsae.dsae import DSAE_Encoder, CustomDeepSpatialAutoencoder
from lib.dsae.dsae_choose_feature import DSAE_ValFeatureChooser
from lib.dsae.dsae_dataset import DSAE_FeatureProviderDataset
from lib.dsae.dsae_feature_provider import FeatureProvider
from lib.dsae.dsae_manager import DSAEManager
from lib.dsae.dsae_networks import TargetVectorDSAE_Decoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--latent", type=int, required=True)
    parser.add_argument("--dsae_path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--training", type=float, default=0.8)
    parser.add_argument("--output_divisor", type=int, default=4)
    parser.add_argument("--is_bop", default="no")
    parser.add_argument("--trials", type=int)
    parse_result = parser.parse_args()

    dataset = parse_result.dataset
    config = dict(
        name=parse_result.name,
        latent_dimension=parse_result.latent,
        dsae_path=parse_result.dsae_path,
        split=[0.8, 0.1, 0.1],
        training=parse_result.training,
        seed=get_seed("random"),
        size=(96, 128),
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        output_divisor=parse_result.output_divisor,
        is_bop=parse_result.is_bop == "yes",
        trials=parse_result.trials
    )

    if config["is_bop"] and config["trials"] is None:
        raise ValueError("Bayesian optimisation requires a number of trials!")

    device = set_up_cuda(config["seed"])
    config["device"] = device

    height, width = config["size"]

    model = CustomDeepSpatialAutoencoder(
        encoder=DSAE_Encoder(
            in_channels=3,
            out_channels=(config["latent_dimension"] * 2, config["latent_dimension"], config["latent_dimension"] // 2),
            strides=(2, 1, 1),
            normalise=True
        ),
        decoder=TargetVectorDSAE_Decoder(
            image_output_size=(height // config["output_divisor"], width // config["output_divisor"]),
            latent_dimension=config["latent_dimension"],
            normalise=True
        )
    )
    model.load_state_dict(DSAEManager.load_state_dict(os.path.join("models/dsae/", config["dsae_path"])))
    model.to(config["device"])
    feature_provider = FeatureProvider(model=model, device=config["device"])

    feature_provider_dataset = DSAE_FeatureProviderDataset(
        feature_provider=feature_provider,
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        input_resize_transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(height, width))
        ]),
        size=(height, width),
        cache=True,
        add_image=True,
        normalise=False
    )

    chooser = DSAE_ValFeatureChooser(
        name=config["name"],
        latent_dimension=config["latent_dimension"],
        feature_provider_dataset=feature_provider_dataset,
        split=config["split"],
        limit_train_coeff=config["training"],
        device=config["device"]
    )

    if config["is_bop"]:
        searcher = CropSizeFeatureSearch(
            name=config["name"],
            latent_dimension=config["latent_dimension"],
            dsae_feature_chooser=chooser,
            # max and min as defaults for AttentionNetworkCoord
        )
        searcher.search(total_trials=config["trials"])
        searcher.save(os.path.join("models/bop_chooser", config["name"]))
    else:
        # latent_dimension // 2, number of trials
        # TODO: do this for multiple crop sizes
        feature_index = chooser.get_best_feature_index()
        print(f"Best feature index {feature_index} for {config['dsae_path']}.")
        chooser.save(os.path.join("models/dsae_chooser", config["name"]))
