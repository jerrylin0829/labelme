class BaseModel:
    def free_resources(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def run(self, **kwargs):
        raise NotImplementedError("Subclasses should implement this method.")