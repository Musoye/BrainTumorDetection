import torch
from torchvision import transforms
from PIL import Image

def evaluate_single_image(model, image_filename,class_labels=None):
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize()])
    try:
        image = Image.open(image_filename).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        input_tensor = input_tensor
    except Exception as e:
        print(f"Error loading or processing the image: {e}")
        return None

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class_idx = probabilities.argmax().item()


    if class_labels:
        return class_labels[predicted_class_idx]
    return predicted_class_idx
