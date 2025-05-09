import cv2
import numpy as np
from skimage import morphology, filters
import tensorflow as tf

def create_brain_mask(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Create a binary mask for brain tissue, excluding skull and background.
    
    Args:
        image: Input image array
        threshold: Threshold value for creating the initial mask
        
    Returns:
        Binary mask of the same size as input image
    """
    # Normalize image to 0-1 range
    normalized = (image - image.min()) / (image.max() - image.min())
    
    # Create initial mask using threshold
    mask = normalized > threshold
    
    # Remove small objects and fill holes
    mask = morphology.remove_small_objects(mask, min_size=100)
    mask = morphology.remove_small_holes(mask, area_threshold=100)
    
    # Find the largest connected component (the brain)
    labels = morphology.label(mask)
    if labels.max() > 0:  # Check if any regions were found
        largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
        mask = labels == largest_label
    
    # Erode slightly to ensure we're inside the brain
    mask = morphology.erosion(mask, morphology.disk(3))
    
    return mask.astype(np.float32)

def apply_brain_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply brain mask to the image, setting non-brain regions to 0.
    """
    return image * mask

def preprocess_for_model(image_path: str, target_size: tuple = (224, 224)) -> tuple:
    """
    Preprocess image for model input, including brain masking.
    
    Returns:
        Tuple of (preprocessed_image, brain_mask)
    """
    # Read and resize image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, target_size)
    
    # Create brain mask
    brain_mask = create_brain_mask(image)
    
    # Apply mask and normalize
    masked_image = apply_brain_mask(image, brain_mask)
    normalized_image = (masked_image - masked_image.min()) / (masked_image.max() - masked_image.min())
    
    # Prepare for model (add channel and batch dimensions)
    model_input = np.expand_dims(normalized_image, axis=-1)
    model_input = np.expand_dims(model_input, axis=0)
    
    return model_input, brain_mask

def generate_masked_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    brain_mask: np.ndarray,
    layer_name: str = 'conv2d_3'
) -> np.ndarray:
    """
    Generate Grad-CAM visualization that respects the brain mask.
    """
    # Get the specified layer's output
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        class_output = predictions[:, 0]
    
    # Calculate gradients
    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradients
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
    
    # Post-process heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match input size
    heatmap = cv2.resize(heatmap, (image.shape[2], image.shape[1]))
    
    # Apply brain mask to heatmap
    masked_heatmap = heatmap * brain_mask
    
    # Normalize the masked heatmap
    if masked_heatmap.max() > 0:
        masked_heatmap = masked_heatmap / masked_heatmap.max()
    
    return masked_heatmap

