import tensorflow as tf
from tf_keras import backend as K
import logging
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms, models
from openai import OpenAI
import base64
from django.http import HttpRequest
from PIL import Image
from rest_framework import viewsets, status,permissions
from rest_framework.decorators import action,api_view,permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from .models import Patient, Analysis,Disease,TrainingImage,TrainingClass, TrainingImage,CustomUser
from .serializers import PatientSerializer, AnalysisSerializer,UserSerializer,  UserRegistrationSerializer
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from .ml_utils import  train_model
from .serializers import DiseaseSerializer
from io import BytesIO
import traceback
from django.http import StreamingHttpResponse
from django.contrib.auth import get_user_model
from rest_framework_simplejwt.views import TokenObtainPairView
from rest_framework_simplejwt.tokens import RefreshToken
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from django.conf import settings
import json
from .utils import send_analysis_completion_email
from .utils import process_image, generate_explanation
logger = logging.getLogger(__name__)

class CustomBatchNormalization(tf.keras.layers.BatchNormalization):
    def __init__(self, **kwargs):
        if 'axis' in kwargs and isinstance(kwargs['axis'], list):
            kwargs['axis'] = kwargs['axis'][0]  # Convert [3] to 3
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        if isinstance(config['axis'], list):
            config['axis'] = config['axis'][0]
        return config

class PatientViewSet(viewsets.ModelViewSet):
    queryset = Patient.objects.all()
    serializer_class = PatientSerializer

    @action(detail=True)
    def analyses(self, request, pk=None):
        patient = self.get_object()
        analyses = Analysis.objects.filter(patient=patient)
        serializer = AnalysisSerializer(analyses, many=True)
        return Response(serializer.data)


@api_view(['POST', 'DELETE'])  # Add DELETE to allowed methods
def update_analysis_notes(request, pk):
    try:
        analysis = Analysis.objects.get(pk=pk)
    except Analysis.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)

    if request.method == 'DELETE':
        analysis.notes = None  # Clear the notes
        analysis.save()
        return Response(status=status.HTTP_204_NO_CONTENT)
    
    if request.method == 'POST' and 'notes' in request.data:
        analysis.notes = request.data['notes']
        analysis.save()
        serializer = AnalysisSerializer(analysis)
        return Response(serializer.data)
    
    return Response({'error': 'Notes field is required'}, status=status.HTTP_400_BAD_REQUEST)
@api_view(['GET', 'POST'])
def disease_list(request):
    if request.method == 'GET':
        diseases = Disease.objects.all()
        serializer = DiseaseSerializer(diseases, many=True)
        return Response(serializer.data)
    
    elif request.method == 'POST':
        serializer = DiseaseSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def upload_model(request, disease_id):
    try:
        disease = Disease.objects.get(pk=disease_id)
    except Disease.DoesNotExist:
        return Response(status=status.HTTP_404_NOT_FOUND)
    
    if 'model_file' not in request.FILES:
        return Response({'error': 'No model file provided'}, status=status.HTTP_400_BAD_REQUEST)
    
    disease.model_file = request.FILES['model_file']
    disease.is_active = True
    disease.save()
    
    serializer = DiseaseSerializer(disease)
    return Response(serializer.data)


class AnalysisViewSet(viewsets.ModelViewSet):
    queryset = Analysis.objects.all()
    serializer_class = AnalysisSerializer

    def preprocess_image(self, image_file, model_type):
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

    def generate_gradcam(self, model, img_tensor, pred_index=None):
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
            return self.fallback_visualization(img_tensor)

    def fallback_visualization(self, img_array):
        # Simple fallback: return a grayscale version of the input image
        gray = tf.image.rgb_to_grayscale(img_array[0])
        return tf.squeeze(gray).numpy()

    def create_heatmap_overlay(self, image, heatmap):
        try:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
            return overlay
        except Exception as e:
            logger.error(f"Error in create_heatmap_overlay: {str(e)}")
            raise
    def fallback_visualization(self, img_tensor):
        # Simple fallback: return a grayscale version of the input image
        gray = torch.mean(img_tensor, dim=1)
        return gray.squeeze().numpy()
    def load_model(self, model_path, model_type):
        try:
            if model_type == 'breast_cancer':
                model = models.resnet18(pretrained=False)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, 2)
            else:  # For brain_tumor and cancer
                model = models.efficientnet_b0(pretrained=False)
                model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(f"Stack trace: ", exc_info=True)
            raise
        
    @action(detail=False, methods=['POST'])
    def predict(self, request):
     try:
        if 'image' not in request.FILES:
            return Response(
                {'error': 'No image provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        patient_id = request.data.get('patient_id')
        if not patient_id:
            return Response(
                {'error': 'Patient ID is required'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            patient = Patient.objects.get(id=patient_id)
        except Patient.DoesNotExist:
            return Response(
                {'error': 'Patient not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )

        image_file = request.FILES['image']
        model_type = request.data.get('model_type', 'brain_tumor')

        model_path = Path(f'api/ml/saved_models/{model_type}_best_model.pth')
        if not model_path.exists():
            return Response(
                {'error': f'Model for {model_type} not found'}, 
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            img_tensor, original_img = self.preprocess_image(image_file, model_type)
            original_img_cv = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)

            model = self.load_model(model_path, model_type)
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                pred_class = torch.argmax(probabilities).item()
                confidence = probabilities[pred_class].item()

            class_names = ['negative', 'positive']
            pred_label = class_names[pred_class]

            heatmap = self.generate_gradcam(model, img_tensor, pred_class)
            overlay = self.create_heatmap_overlay(original_img_cv, heatmap)

            _, buffer = cv2.imencode('.png', overlay)
            visualization_base64 = base64.b64encode(buffer).decode('utf-8')

            # Generate AI explanation
            explanation = generate_explanation(pred_label, confidence, model_type)

            # Create analysis record
            analysis = Analysis.objects.create(
                patient=patient,
                type=model_type,
                prediction=pred_label,
                confidence=confidence,
                visualization=visualization_base64,
                model_accuracy=0.9422 if model_type == 'brain_tumor' else 0.8951 if model_type == 'cancer' else 0.92
            )

            # Prepare and send email
            try:
                # Convert original image to base64 for email
                original_img_buffer = BytesIO()
                original_img.save(original_img_buffer, format='PNG')
                original_img_base64 = base64.b64encode(original_img_buffer.getvalue()).decode()

                # Prepare detailed report
                detailed_report = f"""
                Analysis Type: {model_type}
                Prediction: {pred_label}
                Confidence: {confidence:.2%}
                Model Accuracy: {analysis.model_accuracy:.2%}
                
                AI-Generated Explanation:
                {explanation}
                
                A visualization of the analysis has been attached to this email.
                You can view the full results in the medical analysis system.
                """

                # Send email with both original image and visualization
                if patient.email:
                    send_analysis_completion_email(
                        analysis,
                        detailed_report=detailed_report,
                        image_data=visualization_base64
                    )
                else:
                    logger.info(f"No email sent for analysis {analysis.id}. Patient has no email address.")

            except Exception as e:
                logger.error(f"Failed to send email for analysis {analysis.id}: {str(e)}")
                # Continue with the response even if email fails

            serializer = AnalysisSerializer(analysis)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error in model prediction: {str(e)}")
            logger.error(f"Stack trace: ", exc_info=True)
            return Response(
                {'error': f'Error during analysis: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

     except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        logger.error(f"Stack trace: ", exc_info=True)
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def generate_ai_explanation(request, analysis_id):
    try:
        analysis = Analysis.objects.get(pk=analysis_id)
    except Analysis.DoesNotExist:
        return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)

    try:
        # Generate explanation based on existing analysis data
        explanation = generate_explanation_and_staging(analysis)
        return Response({
            "explanation": explanation
        })
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return Response(
            {"error": "Failed to generate explanation. Please try again."}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
def extract_heatmap_metadata(base64_str):
    try:
        # Remove the data URI prefix if present
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
            
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_str)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
            
        # Convert to grayscale for heatmap analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate metadata
        total_pixels = gray.size
        non_zero_pixels = np.count_nonzero(gray)
        max_intensity = np.max(gray)
        mean_intensity = np.mean(gray)
        
        # Find regions of high intensity (potential areas of interest)
        threshold = np.percentile(gray, 90)  # Top 10% intensity
        high_intensity_mask = gray > threshold
        high_intensity_regions = np.count_nonzero(high_intensity_mask)
        
        # Calculate centroid of high intensity regions
        y_coords, x_coords = np.nonzero(high_intensity_mask)
        if len(x_coords) > 0 and len(y_coords) > 0:
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
        else:
            centroid_x = centroid_y = 0
            
        return {
            "coverage_ratio": non_zero_pixels / total_pixels,
            "max_intensity": float(max_intensity),
            "mean_intensity": float(mean_intensity),
            "high_intensity_ratio": high_intensity_regions / total_pixels,
            "centroid": (float(centroid_x), float(centroid_y)),
            "image_size": image.shape[:2]
        }
    except Exception as e:
        logger.error(f"Error in extract_heatmap_metadata: {str(e)}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return None

client = OpenAI(api_key="include your key here")
@api_view(['POST'])
def generate_explanation_and_staging(request, analysis_id):
    try:
        analysis = Analysis.objects.get(id=analysis_id)
    except Analysis.DoesNotExist:
        return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)

    try:
        # Get the base64 image data and ensure it's properly formatted
        image_data = analysis.visualization
        
        # Extract metadata from the heatmap
        metadata = extract_heatmap_metadata(image_data)
        if not metadata:
            return Response(
                {"error": "Failed to extract image metadata"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # Format the image data for OpenAI API
        image_content = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{image_data}" if not image_data.startswith('data:') else image_data
            }
        }
        
        prompt = f"""
        You are a medical imaging specialist. Please analyze this medical image and provide your expert opinion.
        
        Key Information:
        Type: {analysis.type}
        Initial Prediction: {analysis.prediction}
        Confidence: {analysis.confidence * 100:.2f}%
        
        Heatmap Analysis:
        - Coverage: {metadata['coverage_ratio']*100:.1f}% of the image shows significant activation
        - Maximum Intensity: {metadata['max_intensity']}/255
        - Mean Intensity: {metadata['mean_intensity']:.1f}/255
        - High Intensity Areas: {metadata['high_intensity_ratio']*100:.1f}% of the image
        - Region of Interest Centroid: {metadata['centroid']}
        
        Please provide a comprehensive analysis of the image, including your observations, potential diagnosis, 
        and recommendations. Feel free to structure your response naturally.
        """
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert medical imaging specialist. Provide detailed analysis of medical images, focusing on what you can actually see in the image and the provided metadata."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content
                    ],
                },
            ],
            max_tokens=1000,
        )
        
        analysis_text = completion.choices[0].message.content.strip()
        
        # Update analysis with the full text
        analysis.stage_description = analysis_text
        analysis.save()
        
        # Send email notification
        try:
            send_analysis_completion_email(analysis)
        except Exception as e:
            logger.error("Failed to send email notification: %s", str(e))
        
        return Response({"analysis": analysis_text})
        
    except Exception as e:
        logger.error("Error generating analysis: %s", str(e))
        logger.error("Stack trace: %s", traceback.format_exc())
        return Response(
            {"error": f"Error processing image or generating analysis: {str(e)}"}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    
class TrainingViewSet(viewsets.ViewSet):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    @action(detail=False, methods=['POST'], url_path='upload-images')
    def upload_images(self, request):
        logger.info("Received image upload request")
        
        if not request.FILES:
            return Response(
                {'error': 'No files provided'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        class_key = request.data.get('class')
        class_name = request.data.get('name')
        disease_name = request.data.get('disease_name')
        images = request.FILES.getlist('images')

        if not all([class_key, class_name, disease_name, images]):
            return Response(
                {'error': 'Missing required fields'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Create or get disease
            disease, _ = Disease.objects.get_or_create(
                name=disease_name,
                defaults={'key': disease_name.lower().replace(' ', '_')}
            )

            # Create or get training class
            training_class, _ = TrainingClass.objects.get_or_create(
                disease=disease,
                key=class_key,
                defaults={'name': class_name}
            )

            uploaded_images = []
            for image in images:
                # Validate file type
                if not image.content_type.startswith('image/'):
                    return Response(
                        {'error': f'Invalid file type: {image.content_type}'}, 
                        status=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
                    )

                training_image = TrainingImage.objects.create(
                    training_class=training_class,
                    image=image
                )
                uploaded_images.append(training_image.image.path)

            return Response({
                'message': f'Successfully uploaded {len(images)} images for {class_name}',
                'paths': uploaded_images
            }, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"Image upload failed: {str(e)}", exc_info=True)
            return Response(
                {'error': f'Image upload failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['POST'], url_path='start')
    def start_training(self, request):
        logger.info("Received training request")
        
        disease_name = request.data.get('disease_name')
        training_data = request.data.get('training_data')

        if not disease_name or not training_data:
            return Response(
                {'error': 'Missing disease name or training data'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            disease = Disease.objects.get(name=disease_name)
            
            # Process training data
            processed_training_data = {}
            for class_info in training_data:
                class_key = class_info.get('key')
                if not class_key:
                    return Response(
                        {'error': 'Invalid class information provided'}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                training_class = TrainingClass.objects.get(
                    disease=disease,
                    key=class_key
                )
                
                images = TrainingImage.objects.filter(training_class=training_class)
                if not images:
                    return Response(
                        {'error': f'No training images found for class: {class_key}'}, 
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                processed_training_data[class_key] = [img.image.path for img in images]

            model_path, accuracy, confidence = train_model(
                disease_name=disease_name,
                training_data=processed_training_data
            )

            return Response({
                'modelPath': model_path,
                'accuracy': float(accuracy),
                'confidence': float(confidence)
            }, status=status.HTTP_200_OK)

        except Disease.DoesNotExist:
            return Response(
                {'error': f'Disease not found: {disease_name}'}, 
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return Response(
                {'error': f'Training failed: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=False, methods=['POST', 'GET'], url_path='start/stream')
    def start_training_stream(self, request):
        """Stream training progress using Server-Sent Events"""
        def event_stream():
            try:
                # Send initial message to establish connection
                yield "data: {\"status\": \"connected\"}\n\n"

                disease_name = request.POST.get('disease_name') or request.GET.get('disease_name')
                training_data = json.loads(request.POST.get('training_data', '[]')) if request.POST else json.loads(request.GET.get('training_data', '[]'))

                if not disease_name or not training_data:
                    yield f"data: {json.dumps({'error': 'Missing required data'})}\n\n"
                    return

                disease = Disease.objects.get(name=disease_name)
                
                # Process training data
                processed_training_data = {}
                for class_info in training_data:
                    class_key = class_info.get('key')
                    if not class_key:
                        yield f"data: {json.dumps({'error': 'Invalid class information'})}\n\n"
                        return
                    
                    training_class = TrainingClass.objects.get(
                        disease=disease,
                        key=class_key
                    )
                    
                    images = TrainingImage.objects.filter(training_class=training_class)
                    if not images:
                        yield f"data: {json.dumps({'error': f'No training images found for class: {class_key}'})}\n\n"
                        return
                    
                    processed_training_data[class_key] = [img.image.path for img in images]

                def progress_callback(epoch, total_epochs):
                    return f"data: {json.dumps({'epoch': epoch, 'totalEpochs': total_epochs})}\n\n"

                model_path, accuracy, confidence = train_model(
                    training_data=processed_training_data,
                    disease=disease,
                    progress_callback=progress_callback
                )

                yield f"data: {json.dumps({'completed': True, 'accuracy': float(accuracy), 'confidence': float(confidence)})}\n\n"

            except Disease.DoesNotExist:
                yield f"data: {json.dumps({'error': f'Disease not found: {disease_name}'})}\n\n"
            except Exception as e:
                logger.error(f"Training failed: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream'
        )
        
        # Add required headers for SSE
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Headers'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        
        return response

    def options(self, request, *args, **kwargs):
        """Handle preflight CORS requests"""
        response = Response()
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Allow-Headers'] = '*'
        response['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return response

class UserViewSet(viewsets.ModelViewSet):
    queryset = CustomUser.objects.all()
    serializer_class = UserSerializer
    permission_classes = [permissions.AllowAny]



@api_view(['POST'])
@permission_classes([AllowAny])
def register_user(request):
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        serializer.save()
        return Response({
            "message": "Registration successful"
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

User = get_user_model()

class CustomTokenObtainPairView(TokenObtainPairView):
    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        if response.status_code == status.HTTP_200_OK:
            user = User.objects.get(username=request.data['username'])
            refresh = RefreshToken.for_user(user)
            response.data['refresh'] = str(refresh)
            response.data['access'] = str(refresh.access_token)
            response.data['user'] = {
                'id': user.id,
                'username': user.username,
                'email': user.email,
            }
        return response

@api_view(['POST'])
@permission_classes([AllowAny])
def logout_view(request):
    try:
        refresh_token = request.data["refresh_token"]
        token = RefreshToken(refresh_token)
        token.blacklist()
        return Response(status=status.HTTP_205_RESET_CONTENT)
    except Exception as e:
        return Response(status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
def update_notification_email(request, analysis_id):
    try:
        analysis = Analysis.objects.get(id=analysis_id)
    except Analysis.DoesNotExist:
        return Response({"error": "Analysis not found"}, status=status.HTTP_404_NOT_FOUND)
    
    email = request.data.get('email')
    if email:
        analysis.notification_email = email
        analysis.save()
        return Response({"message": "Notification email updated successfully"})
    else:
        return Response({"error": "Email is required"}, status=status.HTTP_400_BAD_REQUEST)




def send_analysis_email(request, analysis_id):
    try:
        analysis = Analysis.objects.get(id=analysis_id)
    except Analysis.DoesNotExist:
        return Response({'error': 'Analysis not found'}, status=status.HTTP_404_NOT_FOUND)

    if not analysis.patient.email:
        return Response({'error': 'Patient has no email address'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        send_analysis_completion_email(analysis)
        return Response({'message': 'Email sent successfully'}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Failed to send email for analysis {analysis_id}: {str(e)}")
        return Response({'error': 'Failed to send email'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def send_analysis_to_email(request, analysis_id):
    try:
        analysis = Analysis.objects.get(id=analysis_id)
    except Analysis.DoesNotExist:
        return Response({'error': 'Analysis not found'}, status=status.HTTP_404_NOT_FOUND)

    email = request.data.get('email')
    if not email:
        return Response({'error': 'Email address is required'}, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Create a mock request object for generate_explanation_and_staging
        mock_request = HttpRequest()
        mock_request.method = 'POST'
        mock_request.content_type = 'application/json'
        
        # Call generate_explanation_and_staging with the mock request
        explanation_response = generate_explanation_and_staging(mock_request, analysis_id)
        detailed_explanation = explanation_response.data.get('explanation', '')
        
        # Prepare detailed report without "Dear patient"
        detailed_report = f"""
        Analysis Type: {analysis.type}
        Prediction: {analysis.prediction}
        Confidence: {analysis.confidence:.2%}
        Model Accuracy: {analysis.model_accuracy:.2%}
        
        AI-Generated Analysis:
        {detailed_explanation}
        
        A visualization of the analysis has been attached to this email.
        You can view the full results in the medical analysis system.
        """

        # Create a temporary mail message
        message = Mail(
            from_email=settings.DEFAULT_FROM_EMAIL,
            to_emails=email,
            subject=f'Medical Analysis Results',
            html_content=f"""
            <p>Medical Analysis Results:</p>
            <pre>{detailed_report}</pre>
            """
        )

        # Attach the visualization
        if analysis.visualization:
            attached_file = Attachment(
                FileContent(analysis.visualization),
                FileName('analysis_visualization.png'),
                FileType('image/png'),
                Disposition('attachment')
            )
            message.attachment = attached_file

        # Send the email
        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        response = sg.send(message)
        logger.info(
            f"Email sent for analysis {analysis.id} to {email}. Status Code: {response.status_code}"
        )
        
        return Response({'message': 'Email sent successfully'}, status=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Failed to send email for analysis {analysis_id} to {email}: {str(e)}")
        return Response({'error': 'Failed to send email'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
