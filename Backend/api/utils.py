from sendgrid import SendGridAPIClient

from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition

from django.conf import settings
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import base64
import io
from openai import OpenAI

logger = logging.getLogger(__name__)

def load_model(model_type):
    try:
        if model_type == 'breast_cancer':
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
        else:  # For brain_tumor and cancer
            model = models.efficientnet_b0(pretrained=False)
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

        model_path = f'{settings.BASE_DIR}/ml/saved_models/{model_type}_best_model.pth'
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(image_file, model_type):
    try:
        img = Image.open(image_file).convert('L')  # Convert to grayscale
        img = img.resize((224, 224))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),  # Single channel normalization
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        return img_tensor, img
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        raise

def generate_gradcam(model, img_tensor, pred_index=None):
    try:
        model.eval()
        feature_extractor = model.features
        classifier = model.classifier

        features = feature_extractor(img_tensor)
        output = classifier(features.view(features.size(0), -1))

        if pred_index is None:
            pred_index = output.argmax(dim=1).item()

        target = output[0][pred_index]
        target.backward()

        gradients = model.features[-1].grad
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        for i in range(features.shape[1]):
            features[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(features, dim=1).squeeze()
        heatmap = np.maximum(heatmap.detach().cpu().numpy(), 0)
        heatmap /= np.max(heatmap)

        return heatmap
    except Exception as e:
        logger.error(f"Error in generate_gradcam: {str(e)}")
        logger.error(f"Stack trace: ", exc_info=True)
        return fallback_visualization(img_tensor)

def fallback_visualization(img_tensor):
    gray = torch.mean(img_tensor, dim=1)
    return gray.squeeze().numpy()

def create_heatmap_overlay(image, heatmap):
    try:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        return overlay
    except Exception as e:
        logger.error(f"Error in create_heatmap_overlay: {str(e)}")
        raise

def process_image(image_file, model_type):
    try:
        model = load_model(model_type)
        img_tensor, original_img = preprocess_image(image_file, model_type)
        original_img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            pred_class = torch.argmax(probabilities).item()
            confidence = probabilities[pred_class].item()

        class_names = ['negative', 'positive']
        prediction = class_names[pred_class]

        heatmap = generate_gradcam(model, img_tensor, pred_class)
        overlay = create_heatmap_overlay(original_img_cv, heatmap)

        _, buffer = cv2.imencode('.png', overlay)
        visualization_base64 = base64.b64encode(buffer).decode('utf-8')

        return prediction, confidence, visualization_base64
    except Exception as e:
        logger.error(f"Error in process_image: {str(e)}")
        raise

OPENAI_API_KEY = "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"

def send_analysis_completion_email(analysis, detailed_report, image_data):
    if not analysis.patient.email:
        logger.info(f"No patient email set for analysis {analysis.id}")
        return

    message = Mail(
        from_email=settings.DEFAULT_FROM_EMAIL,
        to_emails=analysis.patient.email,
        subject=f'Analysis {analysis.id} Completed',
        html_content=f"""
        <p>Dear {analysis.patient.name},</p>
        <p>Your analysis has been completed. Here are the results:</p>
        <pre>{detailed_report}</pre>
        <p>You can view the full results in the medical analysis system.</p>
        """
    )

    # Attach the image
    if image_data:
        attached_file = Attachment(
            FileContent(image_data),
            FileName('analysis_image.png'),
            FileType('image/png'),
            Disposition('attachment')
        )
        message.attachment = attached_file

    try:
        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        response = sg.send(message)
        logger.info(
            f"Email sent for analysis {analysis.id} to patient {analysis.patient.email}. Status Code: {response.status_code}"
        )
    except Exception as e:
        logger.error(
            f"Error sending email for analysis {analysis.id} to patient {analysis.patient.email}: {str(e)}"
        )
        raise

def generate_explanation(prediction, confidence, analysis_type):
    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""
    As a medical imaging specialist, provide a detailed explanation for a {analysis_type} analysis with the following results:
    
    Prediction: {prediction}
    Confidence: {confidence:.2%}
    
    Please include:
    1. A brief explanation of what the analysis means
    2. Potential next steps or recommendations
    3. Any relevant medical context or implications
    
    Keep the explanation clear and understandable for a patient, while maintaining medical accuracy.
    """

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert medical imaging specialist providing explanations of medical image analysis results to patients."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500
        )
        
        explanation = completion.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        return "We apologize, but we couldn't generate a detailed explanation at this time. Please consult with your healthcare provider for a full interpretation of your results."

