import argparse

from torch.utils.data import DataLoader

from lib.common.utils import get_preprocessing_transforms, set_up_cuda, get_demonstrations, get_loss, get_seed, \
    get_network_param_if_init_from
from lib.cv.controller import TrainingPixelROI, CropDeviationSampler
from lib.cv.dataset import ImageTipVelocitiesDataset
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.meta.mil import MetaImitationLearning
from lib.networks import *
from lib.stn.stn import LocalisationParamRegressor, SpatialTransformerNetwork, STN_SamplingType
from lib.stn.stn_visualise import visualise

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--training", type=float)
    parser.add_argument("--random_std", type=int)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--loss")
    parser.add_argument("--seed", default="random")
    parser.add_argument("--init_from")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--pos_dim")
    parse_result = parser.parse_args()

    loss_params = get_loss(parse_result.loss)
    seed = get_seed(parse_result.seed)
    print("Seed:", seed)

    version = parse_result.version

    if parse_result.size == 32:
        size = (32, 24)
        divisor = 4
    elif parse_result.size == 64:
        size = (64, 48)
        divisor = 2
    elif parse_result.size == 128:
        # only for STN
        size = (128, 96)
        divisor = 1
    else:
        raise ValueError("Invalid size!")

    print("Cropped image size:", size)
    print("Training cropper divisor:", divisor)
    print("Attention network version:", version)

    add_spatial_maps = False
    add_r_map = False
    if version == "V1":
        version = AttentionNetwork_32 if parse_result.size == 32 else AttentionNetwork
    elif version == "V2":
        version = AttentionNetworkV2_32 if parse_result.size == 32 else AttentionNetworkV2
    elif version.lower() == "coord":
        version = AttentionNetworkCoord_32 if parse_result.size == 32 else AttentionNetworkCoord
        add_spatial_maps = True
    elif version.lower() == "tile":
        version = AttentionNetworkTile_32 if parse_result.size == 32 else AttentionNetworkTile
    elif version.lower() == "pos":
        if parse_result.pos_dim is None:
            raise ValueError("Positional encoding dimension 'pos_dim' has to be specified!")
        version = AttentionNetworkPos_32.create(
            parse_result.pos_dim
        ) if parse_result.size == 32 else AttentionNetworkPos.create(parse_result.pos_dim)
        add_spatial_maps = True
    elif version.lower() == "coord_rot":
        version = AttentionNetworkCoordRot_32 if parse_result.size == 32 else AttentionNetworkCoordRot
        add_spatial_maps = True
        add_r_map = True
    elif version.lower() == "coord_se":
        version = AttentionNetworkCoordSE_32 if parse_result.size == 32 else AttentionNetworkCoordSE
        add_spatial_maps = True
    elif version.lower() == "stn":
        # should not crop
        divisor = 1
        localisation_param_regressor = LocalisationParamRegressor(
            add_coord=True,
            scale=parse_result.scale
        )
        model = AttentionNetworkCoordGeneral.create(*size)(*size)
        add_spatial_maps = True

        def spatial_version(_w, _h, size=size):
            return SpatialTransformerNetwork(
                localisation_param_regressor=localisation_param_regressor,
                model=model,
                output_size=size,
                sampling_type=STN_SamplingType.DEFAULT_BILINEAR
            )

        version = spatial_version
        # size is 128, 96 for images, CoordConv takes downsampled version
        size = (128, 96)
    else:
        raise ValueError(f"Attention network version {version} is not available")

    dataset = parse_result.dataset
    print("Dataset: ", dataset)

    # random crop, if required
    crop_deviation_sampler = None
    if parse_result.random_std is not None:
        std = parse_result.random_std
        print(f"Random cropping, std {std}")
        crop_deviation_sampler = CropDeviationSampler(std=std)

    config = dict(
        seed=seed,
        # if pixel cropper is used to decrease size by two in both directions, size has to be decreased accordingly
        # otherwise we would be feeding a higher resolution cropped image
        # we want to supply a cropped image, corresponding exactly to the resolution of that area in the full image
        size=size,
        velocities_csv=f"{dataset}/velocities.csv",
        rotations_csv=f"{dataset}/rotations.csv",
        metadata=f"{dataset}/metadata.json",
        root_dir=dataset,
        initial_pixel_cropper=TrainingPixelROI(
            480 // divisor, 640 // divisor, add_spatial_maps=add_spatial_maps,
            crop_deviation_sampler=crop_deviation_sampler, add_r_map=add_r_map
        ),
        cache_images=False,
        batch_size=32,
        split=[0.8, 0.1, 0.1],
        name=parse_result.name,
        learning_rate=parse_result.learning_rate,
        max_epochs=parse_result.epochs,
        validate_epochs=1,
        save_to_location="models/",
        network_klass=version,
        loss_params=loss_params,
        patience=parse_result.patience,
        init_from=parse_result.init_from
    )

    print("Name:", config["name"])

    device = set_up_cuda(config["seed"])
    preprocessing_transforms, transforms = get_preprocessing_transforms(config["size"], is_coord=add_spatial_maps,
                                                                        add_r_map=add_r_map)

    dataset = ImageTipVelocitiesDataset(
        velocities_csv=config["velocities_csv"],
        rotations_csv=config["rotations_csv"],
        metadata=config["metadata"],
        root_dir=config["root_dir"],
        initial_pixel_cropper=config["initial_pixel_cropper"],
        transform=preprocessing_transforms,
        ignore_cache_if_cropper=device == torch.device("cpu")
    )

    limit_training_coefficient = parse_result.training or 0.8  # all training data
    print("Training coeff limit:", limit_training_coefficient)

    training_demonstrations, val_demonstrations, test_demonstrations = get_demonstrations(dataset, config["split"],
                                                                                          limit_training_coefficient)

    train_data_loader = DataLoader(training_demonstrations, batch_size=config["batch_size"], num_workers=8,
                                   shuffle=True)
    validation_data_loader = DataLoader(val_demonstrations, batch_size=32, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_demonstrations, batch_size=32, num_workers=8, shuffle=True)

    network_param = get_network_param_if_init_from(
        config["init_from"], config,
        MetaImitationLearning.load_best_params(
            f"models/pretraining_test/{config['init_from']}"
        ) if config["init_from"] is not None else None
    )

    tip_velocity_estimator = TipVelocityEstimator(
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        image_size=config["size"],
        patience=config["patience"],
        **network_param,
        # transforms without initial resize, so they can be pickled correctly
        transforms=transforms,
        name=config["name"],
        device=device,
        composite_loss_params=config["loss_params"]
    )

    tip_velocity_estimator.train(
        data_loader=train_data_loader,
        max_epochs=config["max_epochs"],  # or stop early with patience 10
        validate_epochs=config["validate_epochs"],
        val_loader=validation_data_loader,
        test_loader=test_data_loader
    )

    # save_best_model
    if config["max_epochs"] > 0:
        tip_velocity_estimator.save_best_model(config["save_to_location"])
    # tip_velocity_estimator.plot_train_val_losses()

    if parse_result.version.lower() == "stn":
        # visualise test images and their transformations
        visualise(name=config["name"], model=tip_velocity_estimator.get_network(), dataloader=test_data_loader)

