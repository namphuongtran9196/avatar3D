from modules.smplx.model3d import BaseModel

class SMPLX_UVmap(BaseModel):
    """SMPL-X head model"""
    def __init__(self, name='SMPLX_UVmap'): # Add your aguments here
        super(SMPLX_UVmap, self).__init__(name)
        # Create your model here
        self.model1 = None # For example: self.load_model(model_path)
        self.model2 = None # For example: self.load_model(model_path)
    
    # define you custom functions here
    def _preprocess(self, inputs, **kwargs):
        """Preprocesses the inputs.

        Args:
            inputs (tensor/list/tuple): specifies the input to the model.
            **kwargs: additional keyword arguments.
        """
        pass
        # Example
        # image = cv2.imread(inputs)
        # image = image / 255.0
        # return image

    # the inherited predict function is used to call your custom functions
    def predict(self, inputs, **kwargs):
        """Predicts the output of the model given the inputs.

        Args:
            inputs (tensor/list/tuple): specifies the input to the model.
            **kwargs: additional keyword arguments.
        """
        print("Predict from SMPLX_UVmap")
        pass
        # Example
        # image = self._preprocess(inputs)
        # output1 = self.model1(image)
        # output2 = self.model2(image)
        # return output1, output2