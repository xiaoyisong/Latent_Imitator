import numpy as np
import imageio
from PIL import Image
import os
from tqdm import tqdm


def crop_images(inputPath, outputPath, size=128):
    img_list = [f for f in os.listdir(inputPath) if os.path.splitext(f)[1] == ".jpg"]

    nImgs = len(img_list)

    if not os.path.isdir(outputPath):
        os.mkdir(outputPath)

    cx = 121
    cy = 89
    for item in tqdm(img_list):
        path = os.path.join(inputPath, item)

        with open(path, 'rb') as f:
            img = Image.open(f)

            img = np.array(img.convert('RGB'))
            # if (index==1):
            #     print(img.shape)
            img = img[cx - 64 : cx + 64, cy - 64 : cy + 64]

            path = os.path.join(outputPath, item)
            imageio.imwrite(path, img)


if __name__ == "__main__":
    crop_images(
        "./celebA/img_align_celeba", "./celebA/img_align_celeba_crop",
    )

