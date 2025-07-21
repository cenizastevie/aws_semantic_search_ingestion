import os
import torch
import json
import numpy as np

class ModelHandler:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.initialized = False

    def initialize(self, model_dir):
        """
        Load the model from disk
        """
        model_path = os.path.join(model_dir, 'model.pth')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        self.initialized = True
        print("Model loaded successfully")

    def preprocess(self, input_data):
        """
        Preprocess the input data
        """
        # Convert input to appropriate format for your model
        # This is just an example - adjust based on your input format
        if isinstance(input_data, str):
            # If the input is a string (like JSON)
            input_json = json.loads(input_data)
            input_tensor = torch.tensor(input_json['inputs'], dtype=torch.float32).to(self.device)
        else:
            # If the input is binary
            input_tensor = torch.tensor(np.frombuffer(input_data, dtype=np.float32)).to(self.device)
            
        return input_tensor

    def inference(self, input_tensor):
        """
        Run inference with the model
        """
        with torch.no_grad():
            predictions = self.model(input_tensor)
        return predictions

    def postprocess(self, inference_output):
        """
        Post-process the model output
        """
        # Convert output to the desired format
        return json.dumps({
            'predictions': inference_output.cpu().numpy().tolist()
        })
