import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
from torchvision import transforms


def to_np(img):
    img = img.numpy().transpose((1, 2, 0))
    img = img[:, :, :3]
    img = (img + 1) / 2
    img = np.clip(img, 0, 1)
    return img


def draw_line(start, end, img, colour):
    h, w, c = img.shape
    u = torch.tensor([w - 1, h - 1]).int()
    l = torch.tensor([0, 0]).int()
    start = torch.max(torch.min(start, u), l)
    end = torch.max(torch.min(end, u), l)
    rr, cc = line(*tuple(start.flip(0)), *tuple(end.flip(0)))
    rr = np.clip(rr, 0, h - 1)
    cc = np.clip(cc, 0, w - 1)
    img[rr, cc] = colour


def draw_transformation(image_batch, tl, tr, bl, br, same_colour=True):
    b, c, h, w = image_batch.size()
    out = torch.zeros(b, 3, h, w)
    for i, image in enumerate(image_batch):
        np_img = to_np(image)
        h, w, c = np_img.shape
        size_tensor = torch.tensor([w - 1, h - 1]).float()
        tl_i = (tl[i].squeeze(1) * size_tensor).int()
        tr_i = (tr[i].squeeze(1) * size_tensor).int()
        bl_i = (bl[i].squeeze(1) * size_tensor).int()
        br_i = (br[i].squeeze(1) * size_tensor).int()
        colour1 = (1.0, 0.0, 0.0)
        colour2 = (0.0, 1.0, 0.0)
        colour3 = (0.0, 0.0, 1.0)
        colour4 = (0.0, 1.0, 1.0)
        if same_colour:
            colour1 = colour2 = colour3 = colour4 = (1.0, 0.0, 0.0)

        draw_line(tl_i, tr_i, np_img, colour1)
        draw_line(tr_i, br_i, np_img, colour2)
        draw_line(br_i, bl_i, np_img, colour3)
        draw_line(bl_i, tl_i, np_img, colour4)
        out[i] = (torch.from_numpy(np_img).permute(2, 0, 1) * 2) - 1
    return out


def visualise(name, model, dataloader=None, image_batch=None, return_drawn_image=False):
    if dataloader is None and image_batch is None:
        raise ValueError("Dataloader or image batch")
    model.to(torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        if dataloader is not None:
            iterator = iter(dataloader)
            for i in range(8):
                # better test trajectory to visualise it on
                image_batch = next(iterator)["image"].cpu()

        _ = model(image_batch)
        transformation_params = model.transformation_params
        transformed_images = model.transformed_image
        tl = (transformation_params @ torch.tensor([[-1.0], [-1.0], [1.0]]) + 1) / 2
        tr = (transformation_params @ torch.tensor([[1.0], [-1.0], [1.0]]) + 1) / 2
        bl = (transformation_params @ torch.tensor([[-1.0], [1.0], [1.0]]) + 1) / 2
        br = (transformation_params @ torch.tensor([[1.0], [1.0], [1.0]]) + 1) / 2
        image_batch = draw_transformation(image_batch, tl, tr, bl, br)
        if return_drawn_image:
            return image_batch
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
