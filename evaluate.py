import torch
from torchvision import transforms
from PIL import Image

def evaluate_single_image(model, image_filename, class_labels=['no', 'yes']):
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    try:
        image = Image.open(image_filename).convert('RGB')
        input_tensor = preprocess(image).unsqueeze(0)
        input_tensor = input_tensor
    except Exception as e:
        print(f"Error loading or processing the image: {e}")
        return None

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0).tolist()
    
    result = {f"{class_labels[i]}": f"{probabilities[i] * 100:.2f}%" for i in range(len(probabilities))}

    return result
