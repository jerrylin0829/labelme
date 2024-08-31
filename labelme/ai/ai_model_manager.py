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
        """
        Initialize a model by its name, checking for conflicts with other models and releasing them if necessary.
        If the model is already initialized, it returns the existing instance.

        Args:
            model_name (str): The name of the model to initialize.
            **kwargs: Additional keyword arguments specific to the model's initialization.

        Returns:
            object: The initialized model instance, or None if initialization fails.
        """
        conflicting_model_name = self._oom_avoid(model_name)
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
        """
        Build a model instance based on the provided model name and initialization parameters.
        It looks for the model class in the MODELS_ENABLE list and initializes it.

        Args:
            model_name (str): The name of the model to build.
            **kwargs: Additional keyword arguments specific to the model's construction.

        Returns:
            object: The constructed model instance.

        Raises:
            ValueError: If the model name is not found in the MODELS_ENABLE list.
            Exception: If there is an error during model construction.
        """
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
        """
        Retrieve the current model. If the model is not available, it initializes and sets it as the current model.

        Args:
            model_name (str): The name of the model to retrieve.
            **kwargs: Additional keyword arguments specific to the model's initialization.

        Returns:
            object: The current model instance.
        """
        if not self.is_model_available(model_name): 
            self.set_current_model(model_name, **kwargs)
        return self.current_model
    
    def set_current_model(self, model_name, **kwargs):
        """
        Set the specified model as the current model, initializing it if necessary.

        Args:
            model_name (str): The name of the model to set as current.
            **kwargs: Additional keyword arguments specific to the model's initialization.
        """
        self.current_model = self.initialize_model(model_name, **kwargs)
        if self.current_model is None:
            logger.fatal(f"Failed to set model {model_name} as current.")

    def update_default_img(self, img):
        """
        Update the default image used by the manager.

        Args:
            img (np.ndarray): The image to be used as the default image.
        """
        
        self.img = img
        for model_name in self.models.keys():
            self.set_model_img(model_name,img)
            
    def set_model_img(self, model_name, img):
        """
        Set the image for the specified model. Logs the process and handles any errors that may occur.

        Args:
            model_name (str): The name of the model to set the image for.
            img (np.ndarray): The image to set for the model.
        """
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
        """
        Set the image for the current model.

        Args:
            img (np.ndarray): The image to set for the current model.
        """
        img = self.img if img is None else img_qt_to_arr(img)
        if self.current_model:
            self.current_model.setImg(image=img)
        
    def set_inference_dev(self, model_name, dev_num):
        """
        Set the inference device for a specific model.

        Args:
            model_name (str): The name of the model to set the device for.
            dev_num (int): The device number to set for inference.
        """
        model = self.models.get(model_name)
        if model and model.name == "EfficientSam (accuracy)":
            model.set_providers(dev_num)
            
    def is_model_available(self, model_name):
        """
        Check if a specific model is available (i.e., already initialized).

        Args:
            model_name (str): The name of the model to check.

        Returns:
            bool: True if the model is available, False otherwise.
        """
        return model_name in self.models
     
    def release_model(self, model_name):
        """
        Release the resources associated with a specific model and remove it from the model dictionary.

        Args:
            model_name (str): The name of the model to release.
        """
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
        """
        Release all models and their associated resources.
        """
        for model_name in list(self.models.keys()):
            self.release_model(model_name)

    def run_model(self, model_name, **kwargs):
        """
        Run a specific model, initializing it first if it is not available.

        Args:
            model_name (str): The name of the model to run.
            **kwargs: Additional keyword arguments passed to the model's run method.

        Returns:
            object: The result of the model's run method, or None if the model is not available.
        """
        if not self.is_model_available(model_name):
            model = self.initialize_model(model_name, **kwargs)
        else: 
            model = self.models[model_name]
            
        if model:
            return self._run_model_with_recovery(model_name, **kwargs)
        return None

    def run_current_model(self, **kwargs):
        """
        Run the current model.

        Args:
            **kwargs: Additional keyword arguments passed to the model's run method.

        Returns:
            object: The result of the model's run method, or None if no current model is set.
        """
        if self.current_model:
            return self._run_model_with_recovery(self.current_model.name, **kwargs)
        logger.warning("No current model is set.")
        return None

    def _run_model_with_recovery(self, model_name, **kwargs):  
        """
        Run a model with recovery logic in case of a runtime error, particularly for out of memory issues.

        Args:
            model_name (str): The name of the model to run.
            **kwargs: Additional keyword arguments passed to the model's run method.

        Returns:
            object: The result of the model's run method, or the result after recovery attempts.
        """       
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
        """
        Execute the run method of the specified model.

        Args:
            model_name (str): The name of the model to run.
            **kwargs: Additional keyword arguments passed to the model's run method.

        Returns:
            object: The result of the model's run method, or None if the model is not available.
        """
        model = self.models[model_name]
        if not model:
            logger.fatal(f"Model {model_name} is not available.")
            return None
        return model.run(**kwargs)

    def _try_recover_model(self, model_name, bbox, retry_count=0, max_retries=3):
        """
        Attempt to recover from a failure by reducing the grid size and retrying the model execution.
        This is useful for out-of-memory errors.

        Args:
            model_name (str): The name of the model to recover.
            bbox (list): The bounding box parameters for the model's run method.
            retry_count (int, optional): The current retry attempt count. Defaults to 0.
            max_retries (int, optional): The maximum number of retries allowed. Defaults to 3.

        Returns:
            object: The result of the model's run method after recovery, or None if recovery fails.
        """
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
    
    def _oom_avoid(self, model_name):
        """
        Avoiding Out of Memory (OOM) errors
        """
        if model_name == "EfficientSAM_Everything" and "EfficientSam (accuracy)" in self.models:
            return "EfficientSam (accuracy)"
        if model_name == "EfficientSam (accuracy)" and "EfficientSAM_Everything" in self.models:
            return "EfficientSAM_Everything"
        return None
