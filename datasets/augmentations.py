# Basic 3D augmentations with volumentations
import volumentations as volumen

def get_augmentation():
    return volumen.Compose([
        volumen.RandomGamma(gamma_limit=(60, 150), p=0.3),
        volumen.GaussianNoise(var_limit=(0, 5), p=0.3),
        volumen.Flip(1, p=0.3),
        volumen.Flip(2, p=0.3),
    ], p=1.0)
