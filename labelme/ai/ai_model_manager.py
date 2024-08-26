from . import MODELS_ENABLE
from .eSam.build_esam import build_efficient_sam_vits
from labelme.logger import logger
from labelme.utils import img_qt_to_arr
class AIModelManager:
    
    def __init__(self):
        self.models = {}
        self.current_model = None

    def initialize_model(self, model_name, **kwargs):
        if model_name in self.models:
            return self.models[model_name]
        try:
            model = self.build_model(model_name, **kwargs)
            self.models[model_name] = model
            return model
        except Exception as e:
            logger.fatal(f"Error initializing model {model_name}: {e}")
            return None

    def build_model(self, model_name, **kwargs):
        for model_class in MODELS_ENABLE:
            if model_class.name == model_name:
                if model_class.name == "EfficientSAM_Everything":
                    return build_efficient_sam_vits(batch=kwargs.get("batch", 1), dev=kwargs.get("dev"))
                return model_class(**kwargs)
        raise ValueError(f"Unknown model name: {model_name}")

    def set_current_model(self, model_name, **kwargs):
        self.current_model = self.initialize_model(model_name, **kwargs)
        if self.current_model is None:
            logger.fatal(f"Failed to set model {model_name} as current.")
            
    def set_current_model_img(self,img):
        self.model.setImg(
            image = img_qt_to_arr(img)
        )
        
    def release_model(self, model_name):
        if model_name in self.models:
            model = self.models[model_name]
            try:
                model.free_resources()
                del self.models[model_name]
                logger.info(f"Model {model_name} resources have been released.")
            except Exception as e:
                logger.fatal(f"Error releasing model {model_name}: {e}")
        else:
            logger.warning(f"Model {model_name} not found.")

    def release_all_models(self):
        for model_name in list(self.models.keys()):
            self.release_model(model_name)

    def run_model(self, model_name, **kwargs):
        model = self.initialize_model(model_name, **kwargs)
        if model:
            return self._run_model_logic(model, **kwargs)
        return None

    def run_current_model(self, **kwargs):
        if self.current_model:
            return self._run_model_logic(self.current_model, **kwargs)
        logger.warning("No current model is set.")
        return None

    def _run_model_logic(self, model, **kwargs):
        try:
            if model.name == "EfficientSAM_Everything":
                return model.run_in_background(bbox=kwargs.get("bbox"))
            
            if kwargs.get("ai_type") == "ai_polygon":
                return model.predict_polygon_from_points(
                    points=kwargs.get("points"), 
                    point_labels=kwargs.get("point_labels")
                )
            return model.predict_mask_from_points(
                points=kwargs.get("points"), 
                point_labels=kwargs.get("point_labels")
            )
        except Exception as e:
            logger.fatal(f"Error running model {model.name}: {e}")
            return None
