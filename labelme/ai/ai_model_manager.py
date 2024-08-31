import traceback
import torch
import gc
from . import MODELS_ENABLE
from .eSam.build_esam import build_efficient_sam_vits
from labelme.logger import logger
from labelme.utils import img_qt_to_arr
from .eSam.esam_everything import EfficientSAM_Everything

class AIModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.img = None

    def initialize_model(self, model_name, **kwargs):
        conflicting_model_name = self._get_conflicting_model_name(model_name)
        if conflicting_model_name:
            self.release_model(conflicting_model_name)

        img = kwargs.get('img', self.img)
        if model_name in self.models:
            logger.info("Model already initialized.")
            return self.models[model_name]

        try:
            logger.warning(f"Initializing model: {model_name}")
            model = self.build_model(model_name, **kwargs)
            self.models[model_name] = model
            self.set_model_img(model_name, img)   
            logger.info(f"Model {model_name} initialized successfully")
            return model
        except Exception as e:
            logger.fatal(f"Error initializing model {model_name}: {e}")
            logger.fatal(traceback.format_exc())  
            return None

    def build_model(self, model_name, **kwargs):
        model_class = next((m for m in MODELS_ENABLE if m.name == model_name), None)
        logger.debug(model_class)
        if not model_class:
            logger.error(f"Model class not found for model name: {model_name}")
            raise ValueError(f"Unknown model name: {model_name}")

        try:
            if model_class.name == "EfficientSAM_Everything":
                model = build_efficient_sam_vits(batch=kwargs.get("batch", 1), dev=kwargs.get("dev"))
                return EfficientSAM_Everything(
                    model, 
                    dev=kwargs.get("dev"),
                    nms_thresh=kwargs.get("nms_thresh")
                )
            return model_class(**kwargs)
        except Exception as e:
            logger.fatal(f"Error building model {model_name}: {e}")
            raise  

    def get_current_model(self, model_name, **kwargs):
        if not self.is_model_available(model_name): 
            self.set_current_model(model_name, **kwargs)
        return self.current_model
    
    def set_current_model(self, model_name, **kwargs):
        self.current_model = self.initialize_model(model_name, **kwargs)
        if self.current_model is None:
            logger.fatal(f"Failed to set model {model_name} as current.")

    def update_default_img(self, img):
        self.img = img
            
    def set_model_img(self, model_name, img):
        logger.info(f"Setting image for model: {model_name}")
        tmp = img
        logger.debug(f"Before setImg, Model type: {type(self.models[model_name])}")
        try:
            self.models[model_name].setImg(self.img if tmp is None else tmp)
            logger.debug("setImg method was called successfully.")
        except Exception as e:
            logger.fatal(f"Error during setImg: {e}")
        logger.debug(f"After setImg, Model type: {type(self.models[model_name])}")

    def set_current_model_img(self, img):
        img = self.img if img is None else img_qt_to_arr(img)
        if self.current_model:
            self.current_model.setImg(image=img)
        
    def set_inference_dev(self, model_name, dev_num):
        model = self.models.get(model_name)
        if model and model.name == "EfficientSam (accuracy)":
            model.set_providers(dev_num)
            
    def is_model_available(self, model_name):
        return model_name in self.models
     
    def release_model(self, model_name):
        model = self.models.pop(model_name, None)
        if model:
            try:
                model.free_resources()
                logger.info(f"Model {model_name} resources have been released.")
            except Exception as e:
                logger.fatal(f"Error releasing model {model_name}: {e}")
        else:
            logger.warning(f"Model {model_name} not found.")

    def release_all_models(self):
        for model_name in list(self.models.keys()):
            self.release_model(model_name)

    def run_model(self, model_name, **kwargs):
        if not self.is_model_available(model_name):
            model = self.initialize_model(model_name, **kwargs)
        else: 
            model = self.models[model_name]
            
        if model:
            return self._run_model_with_recovery(model_name, **kwargs)
        return None

    def run_current_model(self, **kwargs):
        if self.current_model:
            return self._run_model_with_recovery(self.current_model.name, **kwargs)
        logger.warning("No current model is set.")
        return None

    def _run_model_with_recovery(self, model_name, **kwargs):  
        try:
            return self._run_model(model_name, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return self._try_recover_model(model_name, **kwargs)
            else:
                logger.fatal(f"Runtime error: {e}")
                raise
        except Exception as e:
            logger.fatal(f"Unexpected error: {e}")
            raise

    def _run_model(self, model_name, **kwargs):  
        model = self.models[model_name]
        if not model:
            logger.fatal(f"Model {model_name} is not available.")
            return None
        return model.run(**kwargs)

    def _try_recover_model(self, model_name, bbox, retry_count=0, max_retries=3):
        #! Currently only available in everything mode
        if retry_count >= max_retries:
            logger.fatal("Maximum number of retries exceeded, model cannot be recovered.")
            return None

        try:
            self.models[model_name].setEverythingGrid(int(self.models[model_name].getEverythingGrid() / 2))
            masks = self._run_model(model_name, bbox=bbox)
            return masks
        except Exception as e:
            logger.fatal(f"Model Reconstruction Failure on retry {retry_count + 1}/{max_retries}: {e}")
            return self._try_recover_model(model_name, bbox, retry_count + 1, max_retries)
    
    def _get_conflicting_model_name(self, model_name):
        if model_name == "EfficientSAM_Everything" and "EfficientSam (accuracy)" in self.models:
            return "EfficientSam (accuracy)"
        if model_name == "EfficientSam (accuracy)" and "EfficientSAM_Everything" in self.models:
            return "EfficientSAM_Everything"
        return None
