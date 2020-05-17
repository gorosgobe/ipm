import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


def to_np(img):
    img = img.numpy().transpose((1, 2, 0))
    img = img[:, :, :3]
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    return img


def visualise(name, model, dataloader):
    model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        image_batch = next(iter(dataloader))["image"].cpu()
        _ = model(image_batch)
        transformed_images = model.transformed_image
        input_grid = to_np(torchvision.utils.make_grid(image_batch, padding=8))
        output_grid = to_np(torchvision.utils.make_grid(transformed_images, padding=8))
        f, axarr = plt.subplots(2, 1, figsize=(15, 15))
        axarr[0].imshow(input_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(output_grid)
        axarr[1].set_title('Transformed Images')

    plt.ioff()
    plt.savefig(f"{name}_transformed.png")
    plt.close()
