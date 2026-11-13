"""The 5 ATS image transformations, ported for reproducible generation.

Port of ATS ``gen_data/Dop.py`` (DauOperator) and the parts of
``gen_data/MyImageDataGenerator.py`` it uses (github.com/SSCT-Lab/ATS). DeepGD
generated its test sets with this code. Every operator takes one uint8 image
(H, W, C) in [0, 255] and returns a uint8 image of the same shape.

What changed from ATS, and why:
- Randomness. ATS draws everything from the global NumPy RNG with
  ``seed=None``, so its sets cannot be reproduced. Here every draw comes from a
  per-image ``np.random.Generator``, so files can be regenerated exactly. ATS
  re-seeds the global RNG inside the shear and zoom branches with the same
  ``seed``. With ``seed=None`` that is a no-op (zx and zy are independent);
  with a fixed seed it would force zx == zy. We keep ATS's actual behaviour:
  independent draws.
- Affine warp. ``apply_affine_transform`` is copied from keras_preprocessing
  1.1.2 (what ATS imported, under keras 2.3.1) instead of imported. Later
  ``tf.keras`` versions swap the row/col axes, which mirrors rotation and shear,
  so importing it would silently change the result. Verified bit-identical to
  keras_preprocessing 1.1.2.
- Brightness. ATS calls ``tf.image.random_brightness`` (TF1 eager API). The
  same arithmetic is done in NumPy: uint8 -> [0, 1] float, add delta, scale by
  255.5, clip to [0, 255], truncate to uint8 (``tf.image.convert_image_dtype``
  with saturate=True).
"""
import cv2
import numpy as np
from scipy import ndimage

# Interpolation and edge filling of ATS's MyImageDataGenerator(fill_mode="nearest").
FILL_MODE = "nearest"
CVAL = 0.0
ORDER = 1


def _transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    return np.dot(np.dot(offset_matrix, matrix), reset_matrix)


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode=FILL_MODE, cval=CVAL, order=ORDER):
    """keras_preprocessing.image.apply_affine_transform (angles in degrees)."""
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        transform_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                     [np.sin(theta), np.cos(theta), 0],
                                     [0, 0, 1]])
    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
        transform_matrix = shift_matrix if transform_matrix is None else np.dot(transform_matrix, shift_matrix)
    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        transform_matrix = shear_matrix if transform_matrix is None else np.dot(transform_matrix, shear_matrix)
    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
        transform_matrix = zoom_matrix if transform_matrix is None else np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = _transform_matrix_offset_center(transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndimage.affine_transform(x_channel, final_affine_matrix, final_offset,
                                                   order=order, mode=fill_mode, cval=cval)
                          for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def _signed_range(rng, value_range):
    """ATS's random sign for RT and SR: (lo, hi) or (-hi, -lo) with prob 1/2."""
    lo, hi = value_range
    if rng.choice([True, False]):
        return (-hi, -lo)
    return (lo, hi)


def rotation(x, rotation_range, rng):
    lo, hi = _signed_range(rng, rotation_range)
    return apply_affine_transform(x, theta=rng.uniform(lo, hi))


def shear(x, shear_range, rng):
    lo, hi = _signed_range(rng, shear_range)
    return apply_affine_transform(x, shear=rng.uniform(lo, hi))


def zoom(x, zoom_range, rng):
    (zx_lo, zx_hi), (zy_lo, zy_hi) = zoom_range
    zx = rng.uniform(zx_lo, zx_hi)
    zy = rng.uniform(zy_lo, zy_hi)
    return apply_affine_transform(x, zx=zx, zy=zy)


def brightness(x, max_delta, rng):
    delta = rng.uniform(-max_delta, max_delta)
    flt = x.astype(np.float32) / np.float32(255.0) + np.float32(delta)
    return np.clip(flt * np.float32(255.5), 0, 255).astype(np.uint8)


# ATS my_blur: filter family per mode, one filter drawn uniformly per image.
BLUR_FAMILIES = {
    "hard": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "easy": [0, 1, 4, 9],
    None: [3, 4, 8],
}


def _blur_filter(x, op):
    if op == 0:
        return cv2.blur(x, (2, 2))
    if op == 1:
        return cv2.blur(x, (3, 3))
    if op == 2:
        return cv2.blur(x, (4, 4))
    if op == 3:
        return cv2.blur(x, (5, 5))
    if op == 4:
        return cv2.GaussianBlur(x, (3, 3), 0)
    if op == 5:
        return cv2.GaussianBlur(x, (5, 5), 0)
    if op == 6:
        return cv2.GaussianBlur(x, (7, 7), 0)
    if op == 7:
        return cv2.medianBlur(x, 3)
    if op == 8:
        return cv2.medianBlur(x, 5)
    if op == 9:
        return cv2.bilateralFilter(x, 6, 50, 50)
    raise ValueError(f"Unknown blur op {op}")


def blur(x, mode, rng):
    op = int(rng.choice(BLUR_FAMILIES[mode]))
    return _blur_filter(np.ascontiguousarray(x), op).reshape(x.shape)


OPERATORS = {
    "RT": rotation,
    "ZM": zoom,
    "SR": shear,
    "BR": brightness,
    "BL": blur,
}
