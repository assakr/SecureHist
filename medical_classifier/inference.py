import os
import torch
from torchvision import transforms
from PIL import Image
from .model import CNNModel

class MedicalClassifier:
    """
    Inference class for the Medical CNN model.
    """
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize the classifier by loading the model weights.
        """
        # Define classes (must match the training folder structure alphabetically)
        self.class_names = ['colon_aca', 'colon_n', 'lung_aca']
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # Define transforms
        self.img_height = 180
        self.img_width = 180
        self.transforms = transforms.Compose([
            transforms.Resize((self.img_height, self.img_width)),
            transforms.ToTensor(),
        ])
        
        # Load model
        self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        self.model = CNNModel(num_classes=len(self.class_names))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path: str) -> dict:
        """
        Predict the class of an image.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
            
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            score, predicted_class_idx = torch.max(probabilities, 0)
            
        return {
            "prediction": self.class_names[predicted_class_idx.item()],
            "confidence": float(score.item() * 100),
            "all_probabilities": {
                name: float(prob.item() * 100)
                for name, prob in zip(self.class_names, probabilities)
            }
        }
