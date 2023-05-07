import os
import cv2
from glob import glob
from tqdm import tqdm
from albumentations import HorizontalFlip, GridDistortion, OpticalDistortion, ChannelShuffle, CoarseDropout, CenterCrop, Crop, Rotate


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def augment_data(images, masks, save_path, augment=True):

    for image, mask in tqdm(zip(images, masks), total=len(images)):
        """Extract the Name"""
        name = image.split("\\")[-1].split(".")[0]

        """Read the Image and Mask"""
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask, cv2.IMREAD_COLOR)

        """Augmentation"""
        if augment == True:

            """Horizontal Flip Augmentation"""
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=image, mask=mask)
            image_flip = augmented["image"]
            mask_flip = augmented["mask"]

            """GreyScale Augmentation"""
            image_grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            mask_grey = mask

            """Channel Shuffle Augmentation"""
            aug = ChannelShuffle(p=1)
            augmented = aug(image=image, mask=mask)
            image_channel_shuffle = augmented["image"]
            mask_channel_shuffle = augmented["mask"]

            """Coarse Dropout Augmentation"""
            aug = CoarseDropout(p=1.0, min_holes=3,
                                max_holes=10, max_height=32, max_width=32)
            augmented = aug(image=image, mask=mask)
            image_droupout = augmented["image"]
            mask_droupout = augmented["mask"]

            """Rotate Augmentation"""
            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=image, mask=mask)
            image_rotate = augmented["image"]
            mask_rotate = augmented["mask"]

            X = [image, image_flip, image_grey,
                 image_channel_shuffle, image_droupout, image_rotate]
            Y = [mask, mask_flip, mask_grey,
                 mask_channel_shuffle, mask_droupout, mask_rotate]

        else:
            X = [image]
            Y = [mask]

        index = 0
        for i, m in zip(X, Y):

            temp_image_name = f"{name}_{index}.png"
            temp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "image", temp_image_name)
            mask_path = os.path.join(save_path, "mask", temp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            index += 1


def list_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image_path = os.path.normpath(image_path)
                image_paths.append(image_path)
    return image_paths


if __name__ == "__main__":

    train_Images = list_image_paths('signatures/TrainSet/X')
    train_Masks = list_image_paths('signatures/TrainSet/Y')
    test_Images = list_image_paths('signatures/TestSet/X')
    test_Masks = list_image_paths('signatures/TestSet/Y')

    """Create directories to save the augmented data"""
    create_dir("new_data/train/image/")
    create_dir("new_data/train/mask/")
    create_dir("new_data/test/image/")
    create_dir("new_data/test/mask/")

    """Data Augmentation"""
    augment_data(train_Images, train_Masks,
                 "new_data/train/", augment=True)
    augment_data(test_Images, test_Masks, "new_data/test/", augment=False)
