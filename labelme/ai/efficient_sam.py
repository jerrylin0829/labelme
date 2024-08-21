import collections
import threading

import gc
import imgviz
import numpy as np
import onnxruntime as ort
import skimage

from ..logger import logger
from . import _utils


class EfficientSam:
    def __init__(self, encoder_path, decoder_path):

        self._providers = _utils.get_available_providers()  # added by Alvin

        self._encoder_path = encoder_path  # added by Alvin
        self._decoder_path = decoder_path  # added by Alvin

        self._encoder_session = ort.InferenceSession(self._encoder_path)  # added by Alvin
        self._decoder_session = ort.InferenceSession(self._decoder_path)  # added by Alvin

        self._lock = threading.Lock()
        self._image_embedding_cache = collections.OrderedDict()

        self._thread = None

    def set_providers(self, list_idx):  # added by Alvin
        self._providers = _utils.set_providers(list_idx)
        self._encoder_session = ort.InferenceSession(self._encoder_path, providers=self._providers)
        self._decoder_session = ort.InferenceSession(self._decoder_path, providers=self._providers)
        logger.info("Mode is modified")

    def set_image(self, image: np.ndarray):
        with self._lock:
            self._image = image
            self._image_embedding = self._image_embedding_cache.get(self._image.tobytes())

        if self._image_embedding is None:
            self._thread = threading.Thread(target=self._compute_and_cache_image_embedding)
            self._thread.start()

    def _compute_and_cache_image_embedding(self):
        with self._lock:
            logger.debug("Computing image ...")
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
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        with self._lock:
            return self._image_embedding

    def free_resources(self):
        """ Free up resources including image data, embeddings, and model sessions. """
        with self._lock:
            # Clear image and image embeddings
            self._image = None
            self._image_embedding = None
            self._image_embedding_cache.clear()
            logger.info("Image, image embedding, and cache cleared.")

        if self._encoder_session is not None:
            del self._encoder_session
            self._encoder_session = None
        if self._decoder_session is not None:
            del self._decoder_session
            self._decoder_session = None
        logger.info("Model sessions have been freed.")
        gc.collect()
        logger.info("all resource have been cleared.")
        

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


def _compute_mask_from_points(decoder_session, image, image_embedding, points, point_labels):
    input_point = np.array(points, dtype=np.float32)
    input_label = np.array(point_labels, dtype=np.float32)

    # batch_size, num_queries, num_points, 2
    batched_point_coords = input_point[None, None, :, :]
    # batch_size, num_queries, num_points
    batched_point_labels = input_label[None, None, :]

    decoder_inputs = {
        "image_embeddings": image_embedding,
        "batched_point_coords": batched_point_coords,
        "batched_point_labels": batched_point_labels,
        "orig_im_size": np.array(image.shape[:2], dtype=np.int64),
    }

    masks, _, _ = decoder_session.run(None, decoder_inputs)
    mask = masks[0, 0, 0, :, :]  # (1, 1, 3, H, W) -> (H, W)
    mask = mask > 0.0

    MIN_SIZE_RATIO = 0.05
    skimage.morphology.remove_small_objects(
        mask, min_size=mask.sum() * MIN_SIZE_RATIO, out=mask
    )

    if 0:
        imgviz.io.imsave("mask.jpg", imgviz.label2rgb(mask, imgviz.rgb2gray(image)))
    return mask
