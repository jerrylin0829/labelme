import imgviz
import numpy as np
import onnxruntime as ort
import skimage

from labelme.logger import logger


def get_available_providers(cuda_num=0): ## added by Alvin
    try:
        check_list = ort.get_available_providers()

        if 'CUDAExecutionProvider' in check_list:
            logger.info(f"CUDA {cuda_num} is available and will be used.")
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': cuda_num,
                })]
            return providers
        
    except Exception as e:
        logger.warning(f"Error checking GPU availability: {e}")
    logger.info("CUDAExecutionProvider is not available, using CPUExecutionProvider.")
    
    return ['CPUExecutionProvider']

def set_providers(mode): ## added by Alvin
    if mode[:4] == "CUDA":
        return get_available_providers(int(mode[5:]))
    elif mode[:3] == "CPU":
        logger.info("CPU Mode Selected : CPUExecutionProvider is available and will be used.")
    return ['CPUExecutionProvider']
  
def _get_contour_length(contour):
    contour_start = contour
    contour_end = np.r_[contour[1:], contour[0:1]]
    return np.linalg.norm(contour_end - contour_start, axis=1).sum()

def compute_polygon_from_mask(mask):
    contours = skimage.measure.find_contours(np.pad(mask, pad_width=1))
    if len(contours) == 0:
        logger.warning("No contour found, so returning empty polygon.")
        return np.empty((0, 2), dtype=np.float32)

    contour = max(contours, key=_get_contour_length)
    POLYGON_APPROX_TOLERANCE = 0.004
    polygon = skimage.measure.approximate_polygon(
        coords=contour,
        tolerance=np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE,
    )
    polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
    polygon = polygon[:-1]  # drop last point that is duplicate of first point

    if 0:
        import PIL.Image

        image_pil = PIL.Image.fromarray(imgviz.gray2rgb(imgviz.bool2ubyte(mask)))
        imgviz.draw.line_(image_pil, yx=polygon, fill=(0, 255, 0))
        for point in polygon:
            imgviz.draw.circle_(image_pil, center=point, diameter=10, fill=(0, 255, 0))
        imgviz.io.imsave("contour.jpg", np.asarray(image_pil))

    return polygon[:, ::-1]  # yx -> xy

def compute_mask_mix_polygon(mask):
    # Find contours in the mask
    contours = skimage.measure.find_contours(np.pad(mask, pad_width=1), level=0.5)
    if len(contours) == 0:
        logger.warning("No contour found, so returning empty polygon!")
        return np.empty((0, 2), dtype=np.float32)
    
    polygons = []
    
    for contour in contours:
        POLYGON_APPROX_TOLERANCE = 0.004
        polygon = skimage.measure.approximate_polygon(
            coords=contour,
            tolerance=np.ptp(contour, axis=0).max() * POLYGON_APPROX_TOLERANCE,
        )
        polygon = np.clip(polygon, (0, 0), (mask.shape[0] - 1, mask.shape[1] - 1))
        polygon = polygon[:-1]  # drop last point that is duplicate of first point
        polygons.append(polygon[:, ::-1])  # yx -> xy

    if 0:
        import PIL.Image

        image_pil = PIL.Image.fromarray(imgviz.gray2rgb(imgviz.bool2ubyte(mask)))
        imgviz.draw.line_(image_pil, yx=polygon, fill=(0, 255, 0))
        for point in polygon:
            imgviz.draw.circle_(image_pil, center=point, diameter=10, fill=(0, 255, 0))
        imgviz.io.imsave("contour.jpg", np.asarray(image_pil))

    return polygons
