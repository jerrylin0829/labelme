class BaseModel:
    def free_resources(self):
        raise NotImplementedError("Subclasses should implement this method.")
    def set_img(self, image):
        raise NotImplementedError("Subclasses should implement this method.")
    def set_parameters(self,**kwargs):
        raise NotImplementedError("Subclasses should implement this method.")
    def get_parameters(self,**kwargs):
        raise NotImplementedError("Subclasses should implement this method.")    
    def run(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")