from abc import abstractmethod

class BaseModel(object):
    """Base class for all models"""
    def __init__(self, name='BaseModel'):
        self.name = name
    
    @abstractmethod
    def predict(self, inputs, **kwargs):
        """Predicts the output of the model given the inputs.

        Args:
            inputs (tensor/list/tuple): specifies the input to the model.
            **kwargs: additional keyword arguments.
        """
        pass