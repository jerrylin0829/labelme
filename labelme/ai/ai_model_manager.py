from . import MODELS_ENABLE
from .eSam.build_esam import build_efficient_sam_vits
from labelme.logger import logger
from labelme.utils import img_qt_to_arr


class AIModelManager:
    def __init__(self):
        self.models = {}
        self.current_model = None
        self.img = None

    def initialize_model(self, model_name, **kwargs):
        #! 防止有複數個 model 同時存在，以致爆顯存 
        if model_name == "EfficientSAM_Everything" and "EfficientSam (accuracy)" in self.models:
            self.release_model("EfficientSam (accuracy)")
        if model_name == "EfficientSam (accuracy)" and "EfficientSAM_Everything" in self.models:
            self.release_model("EfficientSAM_Everything")
            
        img = kwargs.get('img', self.img) 

        if model_name in self.models:
            return self.models[model_name]
        try:
            model = self.build_model(model_name, **kwargs)
            self.models[model_name] = model
            self.set_model_img(model_name, img)
            return model
        except Exception as e:
            logger.fatal(f"Error initializing model {model_name}: {e}")
            return None
    
    def build_model(self, model_name, **kwargs):
        model_class = next((m for m in MODELS_ENABLE if m.name == model_name), None)
        if model_class:
            if model_class.name == "EfficientSAM_Everything":
                return build_efficient_sam_vits(batch=kwargs.get("batch", 1), dev=kwargs.get("dev"))
            return model_class(**kwargs)
        raise ValueError(f"Unknown model name: {model_name}")

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
        img = self.img if img == None else img_qt_to_arr(img)
        self.models[model_name].setImg(img)
            
    def set_current_model_img(self, img):
        img = self.img if img == None else img_qt_to_arr(img)
        self.current_model.setImg(image=img)
        
    def set_inference_dev(self, model_name, dev_num):
        if self.models[model_name].name == "EfficientSAM_Everything":
            self.models[model_name].set_providers(dev_num)
            
    def is_model_available(self, model_name):
        return model_name in self.models
     
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
            return self._run_model(model, **kwargs)
        return None

    def run_current_model(self, **kwargs):
        if self.current_model:
            return self._run_model(self.current_model, **kwargs)
        logger.warning("No current model is set.")
        return None

    def _run_model(self, **kwargs):  
        model_name = kwargs.get("model", None)
        ai_type = kwargs.get("ai_type", None)

        if ai_type in ["ai_polygon","ai_mask","ai_boundingbox"]:
            model_name = "EfficientSam (accuracy)" 
            
        try:
            if model_name == "EfficientSAM_Everything":
                return self.models[model_name].run_in_background(bbox=kwargs.get("bbox"))
            
            if ai_type == "ai_polygon":
                return self.models[model_name].predict_polygon_from_points(
                    points=kwargs.get("points"),
                    point_labels=kwargs.get("point_labels")
                )
            return self.models[model_name].predict_mask_from_points(
                points=kwargs.get("points"),
                point_labels=kwargs.get("point_labels")
            )
        except Exception as e:
            logger.fatal(f"Error running model {model_name}: {e}")
            return None
