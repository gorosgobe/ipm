import json
import os

import numpy as np
import torch
from stable_baselines import SAC, PPO2

from bop.crop_size_feature_search import CropSizeFeatureSearch
from lib.common.test_utils import get_testing_configs, get_scene_and_test_scene_configuration, TestConfig
from lib.common.utils import save_image
from lib.cv.controller import TipVelocityController
from lib.cv.tip_velocity_estimator import TipVelocityEstimator
from lib.dsae.dsae import CustomDeepSpatialAutoencoder, DSAE_Encoder
from lib.dsae.dsae_action_predictor import ActionPredictorManager
from lib.dsae.dsae_action_tvec_adapter import DSAETipVelocityEstimatorAdapter
from lib.dsae.dsae_choose_feature import DSAE_ChooserROI, DSAE_ValFeatureChooser
from lib.dsae.dsae_feature_provider import FeatureProvider, FilterSpatialRLFeatureProvider
from lib.dsae.dsae_manager import DSAEManager
from lib.dsae.dsae_networks import TargetVectorDSAE_Decoder
from lib.simulation.camera_robot import CameraRobot
from lib.stn.stn_tvec_adapter import STNControllerAdapter
from meta_networks import MetaAttentionNetworkCoord
from sawyer_camera_adapter import SawyerCameraAdapter, DiscOffsetGenerator
from soft.rnn_tvec_adapter import RNNTipVelocityControllerAdapter
from stn.stn import LocalisationParamRegressor, SpatialTransformerNetwork, SpatialLocalisationRegressor, \
    STN_SamplingType


def get_full_image_networks(scenes_trained, training_list, versions, prefix=""):
    models_res = []
    for s in scenes_trained:
        for v in versions:
            for tr in training_list:
                # models_res.append(f"{prefix}FullImageNetwork_{s}_coord_{tr}_{v}")
                models_res.append(f"{prefix}FullImageNetwork_{s}_{tr}_{v}")

    for s in scenes_trained:
        for vers in versions:
            for tr in training_list:
                for sc in ["32", "64"]:
                    for v in ["a", "coord"]:
                        # models_res.append(f"{prefix}FullImageNetwork_{s}_{v}_{sc}_{tr}_{vers}")
                        pass

    return models_res, TestConfig.FULL_IMAGE


def get_baseline_networks(scenes_trained, training_list, versions, prefix=""):
    models_res = []
    for s in scenes_trained:
        for tr in training_list:
            for v in versions:
                models_res.append(f"{prefix}BaselineNetwork_{s}_{tr}_{v}")
    return models_res, TestConfig.BASELINE


def get_attention_networks(size, scenes_trained, training_list, versions, prefix=""):
    models_res = []
    if size == 64:
        config = TestConfig.ATTENTION_64
    elif size == 32:
        config = TestConfig.ATTENTION_32
    else:
        raise ValueError("Unknown size")

    for s in scenes_trained:
        for ver in versions:
            for tr in training_list:
                for v in ["V1", "V2", "tile"]:
                    if size == 64:
                        models_res.append(f"{prefix}AttentionNetwork{v}_{s}_{tr}_{ver}")
                    else:
                        models_res.append(f"{prefix}AttentionNetwork{v}_{s}_32_{tr}_{ver}")

    return models_res, config


def get_coord_attention_networks(size, scenes_trained, training_list, versions, prefix=""):
    models_res = []
    if size == 64:
        config = TestConfig.ATTENTION_COORD_64
    elif size == 32:
        config = TestConfig.ATTENTION_COORD_32
    else:
        raise ValueError("Unknown size")

    for s in scenes_trained:
        for v in versions:
            for tr in training_list:
                if size == 64:
                    models_res.append(f"{prefix}AttentionNetworkcoord_{s}_{tr}_{v}")
                else:
                    models_res.append(f"{prefix}AttentionNetworkcoord_{s}_32_{tr}_{v}")

    return models_res, config


def get_target_feature_provider_model(dsae_path, latent_dimension, image_output_size=(24, 32)):
    state_dict = DSAEManager.load_state_dict(os.path.join("models", f"{dsae_path}.pt"))
    model = CustomDeepSpatialAutoencoder(
        encoder=DSAE_Encoder(
            in_channels=3,
            out_channels=(latent_dimension * 2, latent_dimension, latent_dimension // 2),
            strides=(2, 1, 1),
            normalise=True
        ),
        decoder=TargetVectorDSAE_Decoder(
            image_output_size=image_output_size,
            latent_dimension=latent_dimension,
            normalise=True
        )
    )
    model.load_state_dict(state_dict)
    return model


def get_dsae_tve_model(dsae_path, action_predictor_path, latent_dimension, k, image_output_size=(24, 32), rl_path=None,
                       is_sac=None, prefix=""):
    if (rl_path is None) + (is_sac is None) == 1:
        raise ValueError("rl_path and is_sac should be both None or both contain information about the RL model")

    model = get_target_feature_provider_model(dsae_path=os.path.join(prefix, dsae_path),
                                              latent_dimension=latent_dimension,
                                              image_output_size=image_output_size)
    device = torch.device("cpu")

    if rl_path is None:
        feature_provider = FeatureProvider(model=model, device=device)
        print("DSAE encoder params", feature_provider.get_num_parameters())
    else:
        path = os.path.join("models/rl", rl_path)
        if is_sac:
            rl_model = SAC.load(path)
        else:
            rl_model = PPO2.load(path)

        feature_provider = FilterSpatialRLFeatureProvider(
            feature_provider_model=model, device=device, rl_model=rl_model, k=k
        )

    action_predictor = ActionPredictorManager.load(
        path=os.path.join("models/", os.path.join(prefix, f"{action_predictor_path}.pt")),
        k=k
    )
    print("Action predictor params", sum([p.numel() for p in action_predictor.parameters()]))

    dsae_tve_model = DSAETipVelocityEstimatorAdapter(feature_provider=feature_provider,
                                                     action_predictor=action_predictor)
    return dsae_tve_model


def get_default_tve_model(tve_prefix, tve_model_name):
    return TipVelocityEstimator.load("models/{}{}.pt".format(tve_prefix, tve_model_name))


def get_recurrent_tve_model(tve_prefix, tve_model_name):
    return RNNTipVelocityControllerAdapter.load(os.path.join("models", tve_prefix, f"{tve_model_name}.pt"))


def get_dsae_chooser_tve_model(tve_prefix, tve_model_name, dsae_path, latent_dimension, is_bop=False, index_path=None,
                               size=32):
    model = get_target_feature_provider_model(dsae_path=dsae_path, latent_dimension=latent_dimension)
    feature_provider = FeatureProvider(model=model, device=torch.device("cpu"))
    print("Feature provider encoder parameters", feature_provider.get_num_parameters())
    if is_bop:
        info = CropSizeFeatureSearch.load_info(f"models/{tve_prefix}{tve_model_name}")
        best_parameters = info["best_parameters"]
        if "feature" not in best_parameters:
            info = DSAE_ValFeatureChooser.load_info(f"models/{index_path}")
            chooser_index = info["index"]
        else:
            chooser_index = best_parameters["feature"]
        # TODO: change this with min/max info
        chooser_crop_size = (
            24 + round((128 - 24) * best_parameters["width"]), 24 + round((96 - 24) * best_parameters["height"])
        )
    else:
        info = DSAE_ValFeatureChooser.load_info(f"models/{tve_prefix}{tve_model_name}")
        chooser_index = info["index"]
        chooser_crop_size = (32, 24) if size == 32 else (64, 48)

    roi_cropper = DSAE_ChooserROI(
        chooser_index=chooser_index,
        chooser_crop_size=chooser_crop_size,
        feature_provider=feature_provider
    )
    tve = TipVelocityEstimator.load(f"models/{tve_prefix}{tve_model_name}_model.pt")
    print("Model parameters", sum([p.numel() for p in tve.network.parameters()]))
    return tve, roi_cropper


def get_stn_tve_model(tve_prefix, tve_model_name, scale, size, sampling_type, dsae=None):
    if dsae is None:
        localisation_param_regressor = LocalisationParamRegressor(
            add_coord=True,
            scale=scale
        )
    else:
        dsae = get_target_feature_provider_model(dsae_path=dsae, latent_dimension=128)
        localisation_param_regressor = SpatialLocalisationRegressor(
            dsae=dsae.encoder,
            latent_dimension=128,
            scale=scale,
            anneal="anneal" in tve_model_name
        )
    model = MetaAttentionNetworkCoord.create(*size)(track=True)

    stn = SpatialTransformerNetwork(
        localisation_param_regressor=localisation_param_regressor,
        model=model,
        output_size=size,
        sampling_type=sampling_type
    )
    print("Parameters", sum([p.numel() for p in stn.parameters()]))

    info = torch.load(os.path.join("models", tve_prefix, f"{tve_model_name}.pt"), map_location=torch.device('cpu'))
    state = info["stn_state_dict"]
    stn.load_state_dict(state)

    return STNControllerAdapter(stn=stn)


def get_dsae_networks(scenes_trained, trains, versions, prefix=""):
    dsae_paths = []
    models = []
    for s in scenes_trained:
        for tr in trains:
            for vs in versions:
                models.append(f"{prefix}act_64_0_1_1_{s}_{tr}_{vs}")
                # during training, we used version 1 of the autoencoder for each action predictor replication
                dsae_paths.append(f"{prefix}target_64_0_1_1_{s}_{tr}_v1")

    return models, TestConfig.DSAE, dsae_paths, 128, 64, None, None


def get_full_recurrent_networks(scenes_trained, trains, versions):
    models = []
    for s in scenes_trained:
        for tr in trains:
            for vs in versions:
                # models.append(f"LSTMNetwork_full_{s}_{tr}_{vs}")
                # models.append(f"LSTMNetwork_fullcoord_{s}_{tr}_{vs}")
                # models.append(f"LSTMNetwork_mask_{s}_{tr}_{vs}")
                models.append(f"LSTMNetwork_context_{s}_{tr}_{vs}")

    return models, TestConfig.RECURRENT_FULL


def get_coordconv_recurrent_networks(scenes_trained, trains, versions):
    models = []
    for s in scenes_trained:
        for tr in trains:
            for vs in versions:
                models.append(F"LSTMNetwork_coordconv32_{s}_{tr}_{vs}")

    return models, TestConfig.RECURRENT_ATTENTION_COORD_32


def get_dsae_chooser_networks(scenes_trained, trains, versions, dsae_prefix, size):
    models = []
    dsae_paths = []
    for s in scenes_trained:
        for tr in trains:
            for v in versions:
                models.append(f"choose_64_0_1_1_{s}_{tr}_{v}" if size == 64 else f"choose_64_0_1_1_32_{s}_{tr}_{v}")
                dsae_paths.append(os.path.join(dsae_prefix, f"target_64_0_1_1_{s}_{tr}_v1"))

    index_paths = [None] * len(models)
    return models, TestConfig.DSAE_CHOOSE, dsae_paths, 128, False, index_paths


def get_bop_chooser_networks(scenes_trained, trains, versions, dsae_prefix):
    models = []
    dsae_paths = []
    index_paths = []
    for s in scenes_trained:
        for tr in trains:
            for v in versions:
                models.append(f"bop_64_0_1_1_32_{s}_{tr}_{v}")
                index_paths.append(f"evaluation/chooser/choose_64_0_1_1_32_{s}_{tr}_{v}")
                dsae_paths.append(os.path.join(dsae_prefix, f"target_64_0_1_1_{s}_{tr}_v1"))

    return models, TestConfig.DSAE_CHOOSE, dsae_paths, 128, True, index_paths


def get_meta_stn_networks(scenes_trained, trains, versions, scale_str, dsae_prefix, stn_type=""):
    sizes = []
    models = []
    dsae_paths = []
    scales = []
    for s in scenes_trained:
        for tr in trains:
            for vs in versions:
                for typ in ["_retrain"]:#["", "_retrain"]:
                    model_name = f"final_meta_stn{stn_type}_040401_{scale_str}_{s}_{tr}_{vs}{typ}"
                    models.append(model_name)
                    if "dsae" not in model_name:
                        dsae_paths.append(None)
                    else:
                        dsae_paths.append(os.path.join(dsae_prefix, f"target_64_0_1_1_{s}_{tr}_v1"))
                    if "_05_" in model_name:
                        scales.append(0.5)
                        sizes.append((64, 48))
                    elif "_025_" in model_name:
                        scales.append(0.25)
                        sizes.append((32, 24))
                    else:
                        raise ValueError("Unknown scale")

    return models, TestConfig.STN, dsae_paths, scales, STN_SamplingType.LINEARISED, sizes

def get_pos_networks(scenes_trained, trains, versions, pos_dim):
    models = []
    for s in scenes_trained:
        for tr in trains:
            for v in versions:
                models.append(f"AttentionNetworkpos{pos_dim}_{s}_32_{tr}_{v}")
    return models, TestConfig.ATTENTION_COORD_32

if __name__ == "__main__":

    trainings = [
        "08"
    ]

    scenes = [
        # "nodist1",
        # "nodist2",
        # "nodist3",
        # "nodist4",
        # "nodist5",
        # "randist1",
        # "randist2",
        # "randist3",
        # "randist4",
        # "randist5"
        "smalldisc",
        # "smalldiscrand"
    ]

    vs = ["v2"]

    prefix = "disc/dsae"
    # models, testing_config_name = get_full_image_networks(scenes, trainings, vs)
    # models, testing_config_name = get_pos_networks(scenes, trainings, vs, pos_dim=5)
    # models, testing_config_name = get_baseline_networks(scenes, trainings, vs)
    # models, testing_config_name = get_attention_networks(64, scenes, trainings, vs)
    # models, testing_config_name = get_coord_attention_networks(32, scenes, trainings, vs)
    models, testing_config_name, dsae_paths, latent_dimension, k, rl_path, is_sac = get_dsae_networks(scenes, trainings, vs)
    # models, testing_config_name = get_coordconv_recurrent_networks(scenes, trainings, vs)
    # models, testing_config_name = get_full_recurrent_networks(scenes, trainings, vs)
    dsae_prefix = "disc/dsae"
    size = 32
    # models, testing_config_name, dsae_paths, latent_dimension, is_bop, index_paths = get_dsae_chooser_networks(
    #     scenes, trainings, vs, dsae_prefix, size=size
    # )

    # models, testing_config_name, dsae_paths, latent_dimension, is_bop, index_paths = get_bop_chooser_networks(
    #     scenes, trainings, vs, dsae_prefix
    # )
    # models, testing_config_name, dsae_paths, scales, sampling_type, sizes = get_meta_stn_networks(
    #     scenes, trainings, vs, "025", dsae_prefix, "_dsae_guided"
    # )
    # for scene in scenes:
    #     for t in trainings:
    #         # for ty in types:
    #         #     for size in sizes:
    #         models.append(f"FullImageNetwork_{scene}_{t}")
    # models = []
    # for v in vs:
    #     models.extend([
    #         f"FullImageNetworkWD{v}_scene1scene1_08",
    #         f"FullImageNetworkWD{v}_scene1scene1_04",
    #         f"FullImageNetworkWD{v}_scene1scene1_02",
    #         f"FullImageNetworkWD{v}_scene1scene1_015",
    #         f"FullImageNetworkWD{v}_scene1scene1_010",
    #         f"FullImageNetworkWD{v}_scene1scene1_005",
    #     ])
    #
    # testing_config_name = TestConfig.ATTENTION_COORD_ROT_32
    # models = [
    #     "AttentionNetworkcoordRot_scene1scene1_32_08",
    #     "AttentionNetworkcoordRot_scene1scene1_32_04",
    #     "AttentionNetworkcoordRot_scene1scene1_32_02",
    #     "AttentionNetworkcoordRot_scene1scene1_32_015",
    #     "AttentionNetworkcoordRot_scene1scene1_32_010",
    #     "AttentionNetworkcoordRot_scene1scene1_32_005",
    #
    # ]
    # testing_config_name = TestConfig.STN
    # prefix = "meta_stn/official/meta_stn_dsae_guided/v1/"
    # models = [
    #     "eval_meta_stn_dsae_guided_05_linearised_scene1scene1_v1_retrain"
    # ]
    # scales = [0.5]
    # sizes = [(64, 48)]
    # sampling_type = STN_SamplingType.LINEARISED,
    # dsae_paths = ["dsae/target_64_0_1_1_08_scene1scene1_v1"]

    # testing_config_name = TestConfig.RECURRENT_BASELINE
    # prefix = "soft_lstm/baseline/v3/"
    # models = [
    #     "lstm_baseline_16_scene1scene1_v3.pt"
    # ]

    # testing_config_name = TestConfig.DSAE_CHOOSE
    # prefix = "bop_chooser/v3/"
    # dsae_path = "target_64_0001_1_1_08_scene1scene1_v1.pt"
    # models = ["bop64_0001_1_1_08_scene1scene1"]
    # latent_dimension = 128
    # is_bop = True
    # index_path = "dsae_chooser/v3/choose_0001_1_1_08_scene1scene1"

    # testing_config_name = TestConfig.DSAE
    # # # define if network is dsae type
    # dsae_path = "target_64_0001_1_1_08_scene1scene1_v1.pt"
    # models = ["act_ppo_dense_k3_64_0001_08_scene1scene1.pt"]
    # latent_dimension = 128
    # k = 3
    # # # for RL based versions
    # # rl_path = "ppo_test_v2"
    # rl_path = "ppo_dense_k3"
    # is_sac = False
    # # is_sac = False
    for model_idx, model_name in enumerate(models):
        print("Model name", model_name)
        s, test, dist_config = get_scene_and_test_scene_configuration(model_name=model_name)
        with s(headless=True, random_distractors=True, distractors_from=dist_config) as (pr, scene):
            # camera_robot = CameraRobot(pr)
            camera_robot = CameraRobot(pr, movable=SawyerCameraAdapter(pr), offset_generator=DiscOffsetGenerator())
            target = scene.get_target()
            break_early = scene.is_break_early()
            testing_configs = get_testing_configs(camera_robot=camera_robot, pixel_target=target.get_pixel_target())
            testing_config = testing_configs[testing_config_name]

            cropper = testing_config["cropper"]
            c_type = testing_config["c_type"]
            if testing_config_name == TestConfig.DSAE:
                print("DSAE-based policy!")
                tve_model = get_dsae_tve_model(
                    prefix=prefix,
                    dsae_path=dsae_paths[model_idx],
                    action_predictor_path=model_name,
                    latent_dimension=latent_dimension,
                    k=k,
                    rl_path=rl_path,
                    is_sac=is_sac
                )
            elif testing_config_name in [TestConfig.RECURRENT_FULL, TestConfig.RECURRENT_ATTENTION_COORD_32,
                                         TestConfig.RECURRENT_BASELINE]:
                print("Recurrent policy!")
                tve_model = get_recurrent_tve_model(prefix, model_name)
            elif testing_config_name == TestConfig.DSAE_CHOOSE:
                # override cropper to be the DSAE chooser ROI estimator
                tve_model, cropper = get_dsae_chooser_tve_model(
                    tve_prefix=prefix,
                    tve_model_name=model_name,
                    dsae_path=dsae_paths[model_idx],
                    latent_dimension=latent_dimension,
                    is_bop=is_bop,
                    index_path=index_paths[model_idx],
                    size=size
                )
            elif testing_config_name == TestConfig.STN:
                tve_model = get_stn_tve_model(
                    tve_prefix=prefix,
                    tve_model_name=model_name,
                    scale=scales[model_idx],
                    size=sizes[model_idx],
                    sampling_type=sampling_type,
                    dsae=dsae_paths[model_idx]
                )
            else:
                print("Default type policy!")
                tve_model = get_default_tve_model(prefix, model_name)
            controller = TipVelocityController(
                tve_model=tve_model,
                target_object=target,
                camera=camera_robot.get_movable_camera(),
                roi_estimator=cropper,
                controller_type=c_type
            )
            try:
                m = controller.get_model().network

                print("Parameters:", sum([p.numel() for p in m.parameters()]))
                print("Model name:", model_name)
                print("Network: ", m)
                print(controller.get_model().test_loss)
            except Exception:
                print("Parameters not directly available for this network...")

            test_name = "{}.test".format(model_name)
            result_json = {"min_distances": {}, "errors": {}, "fixed_steps_distances": {}, }
            if break_early:
                result_json["achieved"] = {}

            # with open(test, "r") as f:
            #     content = f.read()
            #     js_content = json.loads(content)
            #     json_offset_list = js_content["offset"]
            #     distractor_positions_list = js_content["distractor_positions"]

            with open(dist_config, "r") as f:
                content = f.read()
                js_content = json.loads(content)
                json_offset_list = js_content["offset"]

            achieved_count = 0
            count = 0
            for idx, offset in enumerate(json_offset_list):
                print("Offset:", offset)
                # distractor_positions = distractor_positions_list[idx]
                # print("Distractor positions:", distractor_positions)
                controller.start()
                try:
                    result = camera_robot.run_controller_simulation(
                        controller=controller,
                        offset=np.array(offset),
                        target_position=target.get_final_target().get_position(),
                        scene=scene,
                        sim_gt_velocity=scene.get_sim_gt_velocity(),
                        sim_gt_orientation=scene.get_sim_gt_orientation(),
                        fixed_steps=scene.get_steps_per_demonstration(),
                        break_early=break_early,
                    )
                    count += 1
                    # for index, i in enumerate(result["images"]):
                    #     save_image(i, "/home/pablo/Desktop/test-{}image{}.png".format(count, index))
                    # if testing_config_name == TestConfig.RECURRENT_FULL:
                    #     for index, i in enumerate(controller.get_model().get_np_attention_mapped_images(lower_weight=True)):
                    #         save_image(i, "/home/pablo/Desktop/now_{}_attention-{}image{}.png".format(model_name, count, index))
                    # if testing_config_name == TestConfig.STN:
                    #     for index, i in enumerate(controller.get_model().get_images()):
                    #         save_image(i, "/home/pablo/Desktop/meta_stn/FINAL_{}_stn-{}image{}.png".format(model_name, count, index))
                    result_json["min_distances"][str(idx)] = result["min_distance"]
                    result_json["fixed_steps_distances"][str(idx)] = result["fixed_steps_distance"]
                    result_json["errors"][str(idx)] = dict(
                        combined_errors=result["combined_errors"],
                        velocity_errors=result["velocity_errors"],
                        orientation_errors=result["orientation_errors"]
                    )
                    if break_early:
                        result_json["achieved"][str(idx)] = result["achieved"]

                    if result["fixed_steps_distance"] < 0.03 and not break_early:
                        achieved_count += 1
                    elif break_early and result["achieved"]:
                        print("Break early achieved!")
                        achieved_count += 1

                    print("Min distance: ", result["min_distance"])
                    print("Fixed step distance: ", result["fixed_steps_distance"])
                except:
                    result_json["achieved"][str(idx)] = False
                print("Achieved: ", achieved_count / (idx + 1), "{}/{}".format(achieved_count, idx + 1))

            print("Achieved: ", achieved_count / len(json_offset_list))

            with open(test_name, "w") as f:
                f.write(json.dumps(result_json))
