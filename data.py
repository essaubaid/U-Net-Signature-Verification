import os
import cv2


def crop_and_save_image(image_path, output_path):
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    cropped_image = image[h-500:, w-1024:]
    cv2.imwrite(output_path, cropped_image)


def process_images(input_root, output_root):
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(root, file)
                output_path = os.path.join(
                    output_root, os.path.relpath(input_path, input_root))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                crop_and_save_image(input_path, output_path)


if __name__ == "__main__":
    input_root = "bank_check_images"
    output_root = "signatures"
    process_images(input_root, output_root)
