import collections
import threading
import gc
import imgviz
import numpy as np
import onnxruntime as ort
import skimage

from ..logger import logger
from . import _utils
from .base_model import BaseModel

#! Has been revised 
class EfficientSam(BaseModel):
    def __init__(self, encoder_path, decoder_path):

        self._providers = _utils.get_available_providers()

        self._encoder_path = encoder_path
        self._decoder_path = decoder_path

        self._encoder_session = ort.InferenceSession(self._encoder_path)
        self._decoder_session = ort.InferenceSession(self._decoder_path)

        self._semaphore = threading.Semaphore(1)
        self._image_embedding_cache = collections.OrderedDict()

    def set_providers(self, list_idx):
        self._providers = _utils.set_providers(list_idx)
        self._encoder_session = ort.InferenceSession(self._encoder_path, providers=self._providers)
        self._decoder_session = ort.InferenceSession(self._decoder_path, providers=self._providers)
        
        

    def setImg(self, image: np.ndarray):
        
        with self._semaphore:
            self._image = image
            logger.debug(f"Image type: {type(image)}")
            self._image_embedding = self._image_embedding_cache.get(self._image.tobytes())
            logger.debug(f"Image embedding type: {type(self._image_embedding)}")

        if self._image_embedding is None:
            self._compute_and_cache_image_embedding()

    def _compute_and_cache_image_embedding(self):
        with self._semaphore:
            logger.debug("Computing image ...")
            logger.debug(f"Image type: {type(self._image)}")
            image = imgviz.rgba2rgb(self._image)
            batched_images = image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            (self._image_embedding,) = self._encoder_session.run(
                output_names=None,
                input_feed={"batched_images": batched_images},
            )
            if len(self._image_embedding_cache) > 10:
                self._image_embedding_cache.popitem(last=False)
            self._image_embedding_cache[self._image.tobytes()] = self._image_embedding
            logger.debug("Done computing image embedding.")

    def _get_image_embedding(self):
        with self._semaphore:
            return self._image_embedding

    def free_resources(self):
        with self._semaphore:
            self._image = None
            self._image_embedding = None
            self._image_embedding_cache.clear()
            logger.info("Image, image embedding, and cache cleared.")

            if self._encoder_session is not None:
                try:
                    del self._encoder_session
                except Exception as e:
                    logger.error(f"Error in deleting encoder session: {e}")
                finally:
                    self._encoder_session = None
            if self._decoder_session is not None:
                try:
                    del self._decoder_session
                except Exception as e:
                    logger.error(f"Error in deleting decoder session: {e}")
                finally:
                    self._decoder_session = None

            logger.info("Model sessions have been freed.")
            gc.collect()
            logger.info("All resources have been cleared.")
    
    def predict_mask_from_points(self, points, point_labels):
        return _compute_mask_from_points(
            decoder_session=self._decoder_session,
            image=self._image,
            image_embedding=self._get_image_embedding(),
            points=points,
            point_labels=point_labels,
        )

    def predict_polygon_from_points(self, points, point_labels):
        mask = self.predict_mask_from_points(points=points, point_labels=point_labels)
        return _utils.compute_polygon_from_mask(mask=mask)
    
    def run(self, **kwargs) :
        run_type = kwargs.get("ai_type")
        points,point_labels = kwargs.get("points"),kwargs.get("point_labels")
        
        if run_type ==  "ai_polygon":
            return self.predict_polygon_from_points(points,point_labels)
        return self.predict_mask_from_points(points,point_labels)

def _compute_mask_from_points(decoder_session, image, image_embedding, points, point_labels):
    input_point = np.array(points, dtype=np.float32)
    input_label = np.array(point_labels, dtype=np.float32)

    batched_point_coords = input_point[None, None, :, :]
    batched_point_labels = input_label[None, None, :]

    decoder_inputs = {
        "image_embeddings": image_embedding,
        "batched_point_coords": batched_point_coords,
        "batched_point_labels": batched_point_labels,
        "orig_im_size": np.array(image.shape[:2], dtype=np.int64),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    mask = masks[0, 0, 0, :, :]
    mask = mask > 0.0

    MIN_SIZE_RATIO = 0.05
    skimage.morphology.remove_small_objects(
        mask, min_size=mask.sum() * MIN_SIZE_RATIO, out=mask
    )

    return mask
