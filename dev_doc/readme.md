
# Modular Development User Guide

`AIModelManager` is a general framework for managing and running various AI models. It is used to initialize, manage, and execute different types of models, and can be extended with new models to increase system flexibility.  
`CustomDialog` provides an easy-to-use method for creating PyQt user interaction windows.

## Table of Contents

- [Basic Usage of AIModelManager](#basic-usage)
  - [Initializing AIModelManager](#initializing-aimodelmanager)
  - [Initializing AIModelManager Models](#initializing-aimodelmanager-models)
  - [Usage](#usage)
  - [Initializing Models](#initializing-models)
  - [Running Models](#running-models)
- [Extending with New Models](#extending-with-new-models)
  - [Creating a New Model Class](#creating-a-new-model-class)
- [Template-based PyQt Display Window Development](#template-based-pyqt-display-window-development)
- [Only parameters set by config can be used](#Only-parameters-set-by-config-can-be-used)


## Basic Usage of AIModelManager

### Initializing AIModelManager
First, create an instance of `AIModelManager`:
```python
from ai_model_manager import AIModelManager
manager = AIModelManager()
```
The main instance management is located in `canvas.py` within the `Canvas` class:
```python
class Canvas(QtWidgets.QWidget):
...
 def __init__(self, *args, **kwargs):
       from ..ai.ai_model_manager import AIModelManager
        self.ai_manager = AIModelManager()
        ...
```

### Initializing AIModelManager Models
```python
class Canvas(QtWidgets.QWidget):
...
 def initialize_model(self, model_name, **kwargs):
        conflicting_model_name = self._get_conflicting_model_name(model_name)
        if conflicting_model_name:
            self.release_model(conflicting_model_name)
    ...
```

### Initializing Models
The `model_name` argument is required:
```python
class Canvas(QtWidgets.QWidget):
...
    self.canvas.ai_manager.initialize_model(
                model_name=self._selectAiModelComboBox.currentText(),
            )
```

#### Setting Model Input Image
The `model_name` argument is required:
- `img` is optional unless you want to specify a custom input image. Otherwise, the currently loaded image (the one displayed in the UI) is used as the input.
  - When an image is loaded in `canvas`, `AIModelManager` automatically loads it as well.
  ```python
      self.set_model_img(model_name)   
  ```
- If you want to specify `img`, it must be pre-processed using `img_qt_to_arr`:
```python
    img = img_qt_to_arr(img)
    self.set_model_img(model_name, img)   
```

### Running Models
- The `model_name` argument is required.
- `ai_type` is currently used for the `EfficientSam (accuracy)` feature.
  - If not running the `EfficientSam (accuracy)` function, this argument is not needed.
```python
    points = self.ai_manager._run_model(
        model_name="EfficientSam (accuracy)",
        ai_type = "ai_polygon",
        points=[[point.x(), point.y()] for point in drawing_shape.points],
        point_labels=drawing_shape.point_labels,
    )   
```

## Extending with New Models

### Creating a New Model Class
- The new model must inherit from `BaseModel` in `base_model.py` and override the required methods (etc. `free_resources`, `run`, and `setImg`):
```python
class BaseModel:
    def free_resources(self):
        raise NotImplementedError("Subclasses should implement this method.")
    def set_img(self, image):
        raise NotImplementedError("Subclasses should implement this method.")  
    def run(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
        ...
  class newModel(BaseModel):

  ...
```
- Don't forget to add the new model class to the `MODELS_ENABLE` list in `labelme\ai\__init__.py`:
```python
  MODELS_ENABLE = [ ## added by Alvin
      EfficientSamVitS,
      EfficientSAM_Everything,
      newModel # 
  ]
```

### Integrating into AIModelManager
Extend related functionality within `AIModelManager` as needed.
- Models are stored in `self.models`.
- `self.img` is used as the default input. It updates automatically when an image is loaded, so manual updates are unnecessary.
- Modify functions as needed; the following is an example:
  ```python
    def run_model(self, model_name, **kwargs):
    def _try_recover_model(self, model_name, bbox, retry_count=0, max_retries=3):
    def _oom_avoid(self, model_name):
  ```

## Template-based PyQt Display Window Development
The entire window generation process has been integrated, providing a customizable way to generate related Qt objects. To use this, directly extend the `CustomDialog` class. The code is located in `labelme\widgets\qdialog_generator.py`.

### Adding PyQt Windows and Objects: Usage Guide
- Inherit from `CustomDialog`:
  ```python
  from .qdialog_generator import CustomDialog

  class NewObj(CustomDialog):
      def __init__(self, obj_instance):
        ## fields are used to define the objects displayed in the window
        - type: Qt widget type
        - range (Optional): Used for widgets with a `range` attribute
        - default: Used for interactive objects to pass user input to the instance API
          fields = { 
              'val1': {'type': 'QDoubleSpinBox', 'range': [0, 1.0], 'default': obj_instance.getVal1()},
              'val2': {'type': 'QSpinBox', 'range': [0, 100], 'default': obj_instance.getVal2()},
              'val3': {'type': 'QSpinBox', 'range': [0, 100], 'default': obj_instance.getVal3()},},
          }
        # set_help_content is only needed if help display content is required, and show_help=True must be enabled
          super().__init__(title="Parameter Settings", fields=fields,show_help=True)
          self.set_help_content = (
              "val1: help_content"
          )
          self.obj_instance = obj_instance

      def setParameters(self):
          inputs = self.get_inputs()
          self.obj_instance.setVal1(inputs['val1'])
          self.obj_instance.setVal2(inputs['val2'])
          self.obj_instance.setVal3(inputs['val3'])

  ```
## Only parameters set by config can be used
You can set ai related parameters such as `nms`, `batch_size` in `labelmerc`.
- `nms` : `nms` is used to set the nms threshold, in order to set the parameters of `nms` in the algorithm.
- `batch_size` : Used to set the image used for a single batch, basically the larger the value, the larger the VRAM usage.
```md
  default_shape_color: [0, 255, 0]
  shape_color: auto  # null, 'auto', 'manual'
  shift_auto_shape_color: 0
  label_colors: null
  nms: 0.8
  batch_size: 45
```
