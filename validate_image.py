"""
Medical Image Validator
Validates if uploaded images are actual medical scans
"""

import cv2
import numpy as np
from PIL import Image


def analyze_image_properties(image_path):
    """
    Analyze image properties to determine if it's likely a medical scan
    Returns: dict with analysis results
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        return {
            'is_valid': False,
            'reason': 'Could not read image file',
            'confidence': 0
        }
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Calculate various metrics
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    # Calculate contrast (standard deviation of pixel intensities)
    contrast = std_brightness
    
    # Calculate dynamic range
    min_val = np.min(gray)
    max_val = np.max(gray)
    dynamic_range = max_val - min_val
    
    # Calculate histogram entropy (measure of information content)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize
    hist = hist[hist > 0]  # Remove zero values
    entropy = -np.sum(hist * np.log2(hist))
    
    # Check for medical imaging characteristics
    # Medical scans typically have:
    # - Low to medium brightness (not too bright like photos)
    # - Good contrast
    # - Specific entropy levels
    
    # Grayscale dominance check (medical images are usually grayscale-like)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = np.mean(img_hsv[:, :, 1])
    
    # Edge density (medical images have clear edges)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    
    # Calculate overall medical image score
    score = 0
    
    # Brightness check (medical images are typically darker)
    if 20 < mean_brightness < 200:
        score += 30
        if 40 < mean_brightness < 180:
            score += 10
    
    # Contrast check
    if 20 < contrast < 150:
        score += 25
        if 30 < contrast < 120:
            score += 10
    
    # Saturation check (low saturation = more likely grayscale medical image)
    if saturation < 50:
        score += 20
    elif saturation < 80:
        score += 10
    
    # Edge density check (medical images have clear anatomical structures)
    if 0.02 < edge_density < 0.3:
        score += 20
    
    # Entropy check
    if 3 < entropy < 8:
        score += 15
    
    # Normalize score to 0-100
    confidence = min(100, score)
    
    # Determine if valid
    is_valid = confidence >= 50
    
    reason = ""
    if confidence < 30:
        reason = "Image appears to be a regular photo, not a medical scan"
    elif confidence < 50:
        reason = "Image does not have typical medical scan characteristics"
    elif confidence < 70:
        reason = "Image partially matches medical scan patterns"
    else:
        reason = "Image has typical medical scan characteristics"
    
    return {
        'is_valid': is_valid,
        'confidence': confidence,
        'reason': reason,
        'mean_brightness': float(mean_brightness),
        'contrast': float(contrast),
        'saturation': float(saturation),
        'edge_density': float(edge_density),
        'entropy': float(entropy)
    }


def is_medical_image(image_path):
    """
    Quick check if image is likely a medical scan
    Returns: tuple (is_valid, message)
    """
    result = analyze_image_properties(image_path)
    
    if result['is_valid']:
        return True, "Valid medical image detected"
    else:
        return False, result['reason']


def get_image_type_hint(image_path):
    """
    Try to guess the type of medical imaging
    Returns: suggested modality
    """
    result = analyze_image_properties(image_path)
    
    # Different modalities have different characteristics
    if result['mean_brightness'] < 80 and result['contrast'] > 40:
        return "xray"  # X-rays tend to be darker with good contrast
    elif result['edge_density'] > 0.1:
        return "mri"  # MRIs often have more internal structures
    elif result['mean_brightness'] > 100:
        return "ct"  # CT scans tend to be brighter
    else:
        return "xray"  # Default to xray

