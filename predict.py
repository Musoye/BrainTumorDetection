from  evaluate import evaluate_single_image
from retrieve import load_model
from model import BrainTumorClassifier

def predict_mri(image_filename):
    model = BrainTumorClassifier()
    ans = load_model('./set_brain-tumor.pth', model, optimizer=None, scheduler=None)
    return evaluate_single_image(model, image_filename)