
import albumentations as A
from albumentations.pytorch import ToTensorV2

def collate_fn(batch):
    """
    Handles batch creation of images and bounding boxes with varying sizes
    """
    return tuple(zip(*batch))


def get_train_transform():
    """
    Defines training data transforms
    """
    return A.Compose([
        A.Flip(0.5),
        A.RandomRotate90(0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

def get_valid_transform():
    """
    Defines validation data transforms
    """
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })