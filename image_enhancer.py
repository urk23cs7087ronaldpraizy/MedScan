"""
Image Enhancement Module for MedScan AI
Provides various image processing functions to improve scan visualization
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io


def enhance_image(image_path, enhancement_type='auto'):
    """
    Apply enhancement to medical image
    
    Args:
        image_path: Path to the image file
        enhancement_type: Type of enhancement
            - 'auto': Automatic enhancement
            - 'brightness': Adjust brightness
            - 'contrast': Adjust contrast
            - 'edge': Edge enhancement
            - 'histogram': Histogram equalization
            - 'denoise': Noise reduction
    
    Returns:
        Base64 encoded enhanced image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    enhanced = None
    
    if enhancement_type == 'auto':
        enhanced = auto_enhance(img_rgb)
    elif enhancement_type == 'brightness':
        enhanced = adjust_brightness(img_rgb, 1.3)
    elif enhancement_type == 'contrast':
        enhanced = adjust_contrast(img_rgb, 1.5)
    elif enhancement_type == 'edge':
        enhanced = enhance_edges(img_rgb)
    elif enhancement_type == 'histogram':
        enhanced = histogram_equalization(img_rgb)
    elif enhancement_type == 'denoise':
        enhanced = reduce_noise(img_rgb)
    else:
        enhanced = img_rgb
    
    # Convert to base64
    return convert_to_base64(enhanced)


def auto_enhance(img):
    """Apply automatic enhancement for medical images"""
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    
    # Merge and convert back
    lab = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return enhanced


def adjust_brightness(img, factor=1.3):
    """Adjust image brightness"""
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    
    # Increase value (brightness)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    
    # Merge back
    hsv = cv2.merge([h, s, v])
    brightened = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return brightened


def adjust_contrast(img, factor=1.5):
    """Adjust image contrast"""
    # Apply histogram equalization to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    equalized = cv2.equalizeHist(gray)
    
    # Convert back to RGB (all channels same)
    enhanced = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
    
    # Blend with original
    enhanced = cv2.addWeighted(img, 0.5, enhanced, 0.5, 0)
    
    return enhanced


def enhance_edges(img):
    """Enhance edges for better visualization"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply unsharp masking
    blurred = cv2.GaussianBlur(gray, (0, 0), 3.0)
    sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(sharpened, 50, 150)
    
    # Combine edges with original
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    enhanced = cv2.addWeighted(img, 0.7, edges_colored, 0.3, 0)
    
    return enhanced


def histogram_equalization(img):
    """Apply histogram equalization for better contrast"""
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Equalize Y channel
    y = cv2.equalizeHist(y)
    
    # Merge back
    ycrcb = cv2.merge([y, cr, cb])
    equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return equalized


def reduce_noise(img):
    """Reduce noise in image"""
    # Apply non-local means denoising
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    return denoised


def convert_to_base64(img):
    """Convert numpy array to base64 string"""
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    
    # Convert to base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{img_base64}"


def analyze_image_quality(image_path):
    """
    Analyze the quality of a medical image
    
    Returns:
        Dictionary with quality metrics
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'quality': 'unknown', 'score': 0}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    
    # Calculate sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # Calculate signal-to-noise ratio
    noise = contrast / brightness if brightness > 0 else 0
    
    # Overall quality score (0-100)
    score = 0
    
    # Brightness score (good for medical: 80-180)
    if 80 <= brightness <= 180:
        score += 30
    elif 60 <= brightness <= 200:
        score += 20
    else:
        score += 10
    
    # Contrast score (good: 30-100)
    if 30 <= contrast <= 100:
        score += 30
    elif 20 <= contrast <= 120:
        score += 20
    else:
        score += 10
    
    # Sharpness score
    if sharpness > 100:
        score += 25
    elif sharpness > 50:
        score += 15
    else:
        score += 5
    
    # Noise score
    if 0.3 < noise < 1.0:
        score += 15
    else:
        score += 5
    
    # Determine quality label
    if score >= 80:
        quality = 'Excellent'
    elif score >= 60:
        quality = 'Good'
    elif score >= 40:
        quality = 'Fair'
    else:
        quality = 'Poor'
    
    return {
        'quality': quality,
        'score': score,
        'brightness': round(brightness, 2),
        'contrast': round(contrast, 2),
        'sharpness': round(sharpness, 2),
        'noise_ratio': round(noise, 2)
    }


def get_severity_from_prediction(prediction, confidence, all_probabilities):
    """
    Determine severity level based on prediction and confidence
    
    Args:
        prediction: The predicted disease
        confidence: Confidence percentage
        all_probabilities: Dict of all class probabilities
    
    Returns:
        Severity level: Normal, Mild, Moderate, or Severe
    """
    # Normal / healthy predictions
    normal_conditions = ['Normal', 'Not Fractured', 'No Tumor']
    if prediction in normal_conditions:
        if confidence >= 80:
            return 'Normal'
        else:
            return 'Mild'
    
    # For diseases, determine severity based on risk level
    high_risk_conditions = ['Glioma', 'Lung Tumor', 'COVID-19', 'Brain Tumor']
    medium_risk_conditions = ['Meningioma', 'Pituitary', 'Pneumonia', 'Tuberculosis', "Alzheimer's"]
    low_risk_conditions = ['Fractured', 'Bone Fracture']
    
    if prediction in high_risk_conditions:
        if confidence >= 85:
            return 'Severe'
        elif confidence >= 70:
            return 'Moderate'
        else:
            return 'Mild'
    elif prediction in medium_risk_conditions:
        if confidence >= 85:
            return 'Moderate'
        elif confidence >= 70:
            return 'Mild'
        else:
            return 'Mild'
    elif prediction in low_risk_conditions:
        if confidence >= 85:
            return 'Moderate'
        else:
            return 'Mild'
    
    return 'Requires Review'

