import cv2
import numpy as np


def prepare_img(img):
    """
    Prepare the image for the model, i.e. convert to grayscale, apply gamma correction (2.2) and convert back to BGR (since the model expects 3xHxW images)

    Args:
        img (np.array): image to prepare

    Returns:
        np.array: prepared image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max()- img.min()) # Normalize the image
    img = img ** (1/2.2) # Gamma correction
    img = img * 255 # Scale to 8 bits
    img = img.astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Convert back to BGR
    return img

def segment(img, segmenter, device):
    """
    Segment the image using the segmentation model
    
    Args:
        img (np.array): image to segment
        segmenter (YOLO): segmentation model

    Returns:
        np.array: mask of the image
    """

    segmentation = segmenter(img, verbose=False, device = device)
    mask = segmentation[0].masks.data.cpu().numpy()[0]
    mask = mask.astype(np.uint8) * 255
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    return mask

def detect(img, detector, device):
    """
    Detect the patches in the image using the detection model

    Args:
        img (np.array): image to detect
        detector (YOLO): detection model

    Returns:
        list: list of crops and their coordinates
    """
    img = prepare_img(img)

    detection = detector(img, verbose=False, conf = 0.8, device = device)
    bboxes = detection[0].boxes.xyxy.detach().cpu().numpy()

    crops = []
    for bbox in bboxes:
        # Crop the image
        x1, y1, x2, y2 = bbox[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        crops.append([img[y1:y2, x1:x2], [y1,x1]])

    return crops

