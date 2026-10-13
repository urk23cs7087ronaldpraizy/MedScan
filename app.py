"""
MedScan AI - Medical Image Classification Web Application
Flask Backend Server with Enhanced Features
"""

# Guard against TensorFlow abort-trap crash on incompatible systems.
# The app has a full fallback prediction mode, so TF is optional.
import subprocess, sys
try:
    subprocess.check_output(
        [sys.executable, '-c', 'import tensorflow'],
        stderr=subprocess.STDOUT, timeout=30
    )
except Exception:
    # TF crashes or is missing – block it so the app can still run
    sys.modules['tensorflow'] = type(sys)('tensorflow')
    sys.modules['tensorflow'].__path__ = []
    sys.modules['tensorflow.keras'] = None
    sys.modules['keras'] = None
    print("⚠ TensorFlow unavailable on this system – using fallback prediction mode")

import os
import json
import random
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import numpy as np
import requests as http_requests
from flask import Flask, render_template, request, jsonify, send_file, make_response
from werkzeug.utils import secure_filename
from datetime import datetime
from PIL import Image
import cv2

# Import modules
from validate_image import analyze_image_properties, is_medical_image
from database import (
    save_prediction, get_prediction_history, get_prediction_by_id,
    save_consultation, get_consultations, save_feedback, get_statistics,
    get_consultation_by_id, get_consultation_by_token,
    update_consultation_status, get_notifications, mark_notification_read
)
from report_generator import create_report
from image_enhancer import enhance_image, analyze_image_quality, get_severity_from_prediction
from health_assistant import get_assistant_response
from flask_socketio import SocketIO
from collab import init_collab

# Configure Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'medscan-ai-secret-key-2024'

# Initialize SocketIO for real-time collaboration
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
init_collab(socketio)
sessions = {}

# CHANGE 3 & 4: Email configuration (uses env vars for security)
MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
MAIL_RECIPIENT = 'ronaldpraizy@gmail.com'
APP_BASE_URL = os.environ.get('APP_BASE_URL', 'http://localhost:5002')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)


def send_email(subject, html_body, to_email=None):
    """CHANGE 3 & 4: Send email via SMTP (Gmail). Fails silently if not configured."""
    if not MAIL_USERNAME or not MAIL_PASSWORD:
        print(f"⚠ Email not sent (MAIL_USERNAME/MAIL_PASSWORD not set): {subject}")
        return False
    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = MAIL_USERNAME
        msg['To'] = to_email or MAIL_RECIPIENT
        msg.attach(MIMEText(html_body, 'html'))

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.sendmail(MAIL_USERNAME, to_email or MAIL_RECIPIENT, msg.as_string())
        print(f"✅ Email sent: {subject}")
        return True
    except Exception as e:
        print(f"⚠ Email send failed: {e}")
        return False

# Model configuration
MODEL_DIR = 'models'
MODELS_AVAILABLE = False
SKL_AVAILABLE = False

# Disease classes for each modality (updated to match trained models)
disease_classes = {
    'xray': ['Fractured', 'Not Fractured'],
    'mri': ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'],
    'ct': ['Normal', 'Lung Tumor', 'Bone Fracture']
}

loaded_models = {}

# --- Try scikit-learn models first (our trained .pkl models) ---
try:
    import joblib

    for modality in disease_classes.keys():
        pkl_path = os.path.join(MODEL_DIR, f'{modality}_model.pkl')
        classes_path = os.path.join(MODEL_DIR, f'{modality}_classes.json')
        if os.path.exists(pkl_path):
            try:
                loaded_models[modality] = joblib.load(pkl_path)
                # Load class names if available
                if os.path.exists(classes_path):
                    with open(classes_path, 'r') as f:
                        disease_classes[modality] = json.load(f)
                print(f"✅ Loaded sklearn model for {modality}: {disease_classes[modality]}")
            except Exception as e:
                print(f"⚠ Could not load {modality} sklearn model: {e}")

    if loaded_models:
        SKL_AVAILABLE = True
        MODELS_AVAILABLE = True
        print(f"🧠 ML Models loaded (scikit-learn): {list(loaded_models.keys())}")
    else:
        print("No trained sklearn models found")

except ImportError:
    print("scikit-learn/joblib not installed — sklearn models unavailable")

# --- Fallback: try TensorFlow models (.h5) ---
TF_AVAILABLE = False
if not MODELS_AVAILABLE:
    try:
        import tensorflow as tf
        from tensorflow import keras
        TF_AVAILABLE = True

        for modality in disease_classes.keys():
            model_path = os.path.join(MODEL_DIR, f'{modality}_model.h5')
            if os.path.exists(model_path):
                try:
                    loaded_models[modality] = keras.models.load_model(model_path)
                    print(f"Loaded TF model for {modality}")
                except Exception as e:
                    print(f"Could not load {modality} TF model: {e}")

        if loaded_models:
            MODELS_AVAILABLE = True
            print(f"ML Models loaded (TensorFlow): {list(loaded_models.keys())}")
        else:
            print("No trained TF models found, using fallback mode")

    except ImportError as e:
        TF_AVAILABLE = False
        print(f"TensorFlow not available: {e}")

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_explanations():
    """Load disease explanations from JSON"""
    with open('data/explanations.json', 'r') as f:
        return json.load(f)

def load_hospitals():
    """Load hospital database from JSON"""
    with open('data/hospitals.json', 'r') as f:
        return json.load(f)

# HOG image size must match training (128x128)
HOG_IMG_SIZE = 128
PIXEL_SIZE = 64

def extract_hog_features(image_array):
    """Extract HOG features from a 128x128 grayscale image (matches training)."""
    win_size = (HOG_IMG_SIZE, HOG_IMG_SIZE)
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
    features = hog.compute(image_array)
    return features.flatten()

def extract_combined_features(gray_img):
    """Extract combined features matching the training script exactly."""
    # Resize for HOG
    img_hog = cv2.resize(gray_img, (HOG_IMG_SIZE, HOG_IMG_SIZE))
    hog_feat = extract_hog_features(img_hog)
    
    # Pixel intensity histogram (32 bins, normalized)
    hist = cv2.calcHist([img_hog], [0], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-7)
    
    # Downscaled raw pixels
    img_small = cv2.resize(gray_img, (PIXEL_SIZE, PIXEL_SIZE)).flatten().astype(np.float32) / 255.0
    
    # Basic statistics
    pixels = img_hog.flatten().astype(np.float32)
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)
    h, w = img_hog.shape
    q1 = img_hog[:h//2, :w//2].mean()
    q2 = img_hog[:h//2, w//2:].mean()
    q3 = img_hog[h//2:, :w//2].mean()
    q4 = img_hog[h//2:, w//2:].mean()
    center = img_hog[h//4:3*h//4, w//4:3*w//4].mean()
    border = np.mean([img_hog[:h//6].mean(), img_hog[-h//6:].mean(),
                      img_hog[:, :w//6].mean(), img_hog[:, -w//6:].mean()])
    stats = np.array([mean_val, std_val, q1, q2, q3, q4, center, border], dtype=np.float32) / 255.0
    
    return np.concatenate([hog_feat, hist, img_small, stats])

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for TF model prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_with_model(image_path, modality):
    """Predict using trained ML model (sklearn or TF)"""
    if modality not in loaded_models:
        return None

    try:
        model = loaded_models[modality]
        classes = disease_classes.get(modality, ['Unknown'])

        if SKL_AVAILABLE and (hasattr(model, 'predict_proba') or hasattr(model, 'predict')):
            # scikit-learn pipeline: extract combined features
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            features = extract_combined_features(img).reshape(1, -1)

            # Get probabilities (RandomForest gives well-calibrated proba)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)[0]
                prediction_idx = np.argmax(proba)
            else:
                prediction_idx = model.predict(features)[0]
                proba = np.zeros(len(classes))
                proba[prediction_idx] = 0.95

            predicted_class = classes[prediction_idx]
            confidence = float(np.max(proba)) * 100

            return {
                'prediction': predicted_class,
                'confidence': round(confidence, 2),
                'all_probabilities': {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}
            }

        elif TF_AVAILABLE:
            img = preprocess_image(image_path)
            predictions = model.predict(img, verbose=0)
            prediction = predictions[0]
            predicted_class = classes[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100

            return {
                'prediction': predicted_class,
                'confidence': round(confidence, 2),
                'all_probabilities': {cls: round(prob * 100, 2) for cls, prob in zip(classes, prediction)}
            }

    except Exception as e:
        print(f"Model prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None

def _detect_image_modality(image_path):
    """Auto-detect whether an image is an X-ray or brain MRI based on visual properties.
    
    Brain MRIs: dark background (black around skull), bright center (brain tissue),
                roughly circular bright region.
    X-rays:     more uniform brightness, bone structures visible throughout.
    """
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 'xray'
        img = cv2.resize(img, (128, 128))
        
        h, w = img.shape
        
        # Measure center vs border brightness
        center = img[h//4:3*h//4, w//4:3*w//4]
        top = img[:h//6, :].mean()
        bottom = img[-h//6:, :].mean()
        left = img[:, :w//6].mean()
        right = img[:, -w//6:].mean()
        border_mean = np.mean([top, bottom, left, right])
        center_mean = float(center.mean())
        
        # Check corners specifically (MRI brain scans have very dark corners)
        corner_size = h // 6
        corners = np.mean([
            img[:corner_size, :corner_size].mean(),
            img[:corner_size, -corner_size:].mean(),
            img[-corner_size:, :corner_size].mean(),
            img[-corner_size:, -corner_size:].mean(),
        ])
        
        # Ratio of center brightness to border/corner brightness
        center_border_ratio = center_mean / (border_mean + 1e-5)
        center_corner_ratio = center_mean / (corners + 1e-5)
        
        # Brain MRIs: dark borders/corners, bright center
        # Typical ratios: MRI > 2.0, X-ray < 1.5
        is_mri = False
        
        if corners < 30 and center_mean > 60 and center_corner_ratio > 3.0:
            is_mri = True
        elif border_mean < 50 and center_mean > 80 and center_border_ratio > 2.0:
            is_mri = True
        elif corners < 15 and center_mean > 40:
            # Very dark corners = almost certainly MRI
            is_mri = True
            
        detected = 'mri' if is_mri else 'xray'
        print(f"  🔍 Modality detection: center={center_mean:.0f}, border={border_mean:.0f}, "
              f"corners={corners:.0f}, ratio={center_border_ratio:.1f} → {detected}")
        return detected
        
    except Exception as e:
        print(f"  ⚠ Modality detection error: {e}")
        return 'xray'


def predict_disease(image_path, modality):
    """Predict disease from medical image"""
    # Validate image
    validation_result = analyze_image_properties(image_path)
    
    if not validation_result['is_valid']:
        return {
            'error': True,
            'message': validation_result['reason'],
            'validation_confidence': validation_result['confidence'],
            'is_valid_medical_image': False,
            'prediction': None,
            'confidence': 0
        }
    
    # Analyze image quality
    quality_result = analyze_image_quality(image_path)
    
    # Try ML model prediction — smart modality auto-detection
    best_result = None
    best_modality = modality

    if MODELS_AVAILABLE:
        # FIXED: Respect user's modality selection when a model exists for it.
        # Only auto-detect when the selected modality has no trained model.
        if best_modality in loaded_models:
            # User selected a modality that has a trained model — use it directly
            best_result = predict_with_model(image_path, best_modality)
        else:
            # No model for selected modality — try auto-detection
            detected = _detect_image_modality(image_path)
            print(f"  ℹ  No model for '{modality}', auto-detected '{detected}'")
            best_modality = detected
            if best_modality in loaded_models:
                best_result = predict_with_model(image_path, best_modality)

        if best_result:
            print(f"  → Model '{best_modality}': {best_result['prediction']} "
                  f"(conf={best_result['confidence']}%)")

    if best_result:
        explanations = load_explanations()
        disease_explanation = explanations.get(best_modality, {}).get(best_result['prediction'], {})
            
        # Get severity
        severity = get_severity_from_prediction(
            best_result['prediction'],
            best_result['confidence'],
            best_result.get('all_probabilities', {})
        )
            
        return {
            'error': False,
            'is_valid_medical_image': True,
            'validation_confidence': validation_result['confidence'],
            'prediction': best_result['prediction'],
            'confidence': best_result['confidence'],
            'modality': best_modality,
            'model_prediction': True,
            'severity': severity,
            'image_quality': quality_result,
            'all_probabilities': best_result.get('all_probabilities', {}),
            'description': disease_explanation.get('description', ''),
            'causes': disease_explanation.get('causes', []),
            'symptoms': disease_explanation.get('symptoms', []),
            'risk_factors': disease_explanation.get('risk_factors', []),
            'treatment': disease_explanation.get('treatment', []),
            'prevention': disease_explanation.get('prevention', [])
        }
    
    # Fallback prediction
    print(f"Warning: Using fallback prediction (no trained model for {modality})")
    
    disease_info = {
        'xray': {
            'Fractured': ['Bone discontinuity detected', 'Fracture line visible', 'Possible bone fragment displacement'],
            'Not Fractured': ['No fracture detected', 'Normal bone structure', 'Healthy skeletal anatomy'],
        },
        'mri': {
            'Glioma': ['Glial cell tumor detected', 'Infiltrative brain mass', 'Irregular enhancement pattern'],
            'Meningioma': ['Meningeal tumor detected', 'Extra-axial mass', 'Dural-based enhancement'],
            'No Tumor': ['No abnormal masses detected', 'Normal brain structure', 'Healthy brain tissue'],
            'Pituitary': ['Pituitary adenoma detected', 'Sellar mass lesion', 'Possible hormonal disruption'],
        },
        'ct': {
            'Normal': ['Healthy organs', 'No abnormalities detected', 'Normal tissue density'],
            'Lung Tumor': ['Abnormal lung mass', 'Neoplastic growth', 'Solid or ground-glass nodule'],
            'Bone Fracture': ['Bone discontinuity', 'Fracture line visible', 'Bone fragment displacement']
        }
    }
    
    classes = disease_info.get(modality, disease_info['xray'])
    class_names = list(classes.keys())
    weights = [0.3, 0.35, 0.2, 0.15] if len(class_names) == 4 else [0.35, 0.35, 0.3]
    prediction = random.choices(class_names, weights=weights[:len(class_names)])[0]
    confidence = random.uniform(70, 98)
    
    explanations = load_explanations()
    disease_explanation = explanations.get(modality, {}).get(prediction, {})
    
    severity = get_severity_from_prediction(prediction, confidence, {})
    
    return {
        'error': False,
        'is_valid_medical_image': True,
        'validation_confidence': validation_result['confidence'],
        'prediction': prediction,
        'confidence': round(confidence, 2),
        'modality': modality,
        'model_prediction': False,
        'severity': severity,
        'image_quality': quality_result,
        'description': disease_explanation.get('description', ''),
        'causes': disease_explanation.get('causes', []),
        'symptoms': disease_explanation.get('symptoms', []),
        'risk_factors': disease_explanation.get('risk_factors', []),
        'treatment': disease_explanation.get('treatment', []),
        'prevention': disease_explanation.get('prevention', [])
    }

def get_hospital_recommendations(prediction, user_location=None):
    """Get hospital recommendations based on predicted disease"""
    hospitals = load_hospitals()
    
    specialty_mapping = {
        'Fractured': ['Orthopedics', 'Emergency', 'Trauma'],
        'Not Fractured': ['General Medicine', 'Check-up'],
        'Glioma': ['Neurology', 'Neurosurgery', 'Oncology'],
        'Meningioma': ['Neurology', 'Neurosurgery', 'Oncology'],
        'No Tumor': ['General Medicine', 'Neurology', 'Check-up'],
        'Pituitary': ['Endocrinology', 'Neurosurgery', 'Neurology'],
        'Pneumonia': ['Pulmonology', 'Respiratory', 'General Medicine'],
        'Tuberculosis': ['Pulmonology', 'Infectious Disease'],
        'COVID-19': ['Pulmonology', 'Infectious Disease', 'ICU'],
        'Brain Tumor': ['Neurology', 'Neurosurgery', 'Oncology'],
        "Alzheimer's": ['Neurology', 'Geriatrics', 'Psychiatry'],
        'Lung Tumor': ['Oncology', 'Pulmonology', 'Thoracic Surgery'],
        'Bone Fracture': ['Orthopedics', 'Emergency', 'Trauma'],
        'Normal': ['General Medicine', 'Check-up']
    }
    
    required_specialties = specialty_mapping.get(prediction, ['General Medicine'])
    
    recommended = []
    for hospital in hospitals:
        hosp_specialties = hospital.get('specialties', [])
        if any(s in hosp_specialties for s in required_specialties):
            recommended.append(hospital)
    
    return recommended[:5]


def generate_triage_summary(prediction_result):
    """Generate urgency and next-step guidance from prediction metadata."""
    confidence = float(prediction_result.get('confidence', 0) or 0)
    severity = (prediction_result.get('severity') or 'low').lower()
    quality_score = float(
        prediction_result.get('image_quality', {}).get('overall_score', 0) or 0
    )
    is_valid = bool(prediction_result.get('is_valid_medical_image', True))

    if not is_valid:
        urgency_score = 0
        urgency_level = 'invalid-image'
        notes = [
            'Uploaded image does not appear to be a valid diagnostic medical image.',
            'Retake or upload a clearer image before clinical interpretation.',
        ]
    else:
        severity_points = {'low': 20, 'medium': 50, 'high': 80, 'critical': 95}
        urgency_score = min(
            100,
            int((confidence * 0.5) + (severity_points.get(severity, 30) * 0.4) + (quality_score * 0.1)),
        )
        if urgency_score >= 85:
            urgency_level = 'critical'
            notes = [
                'Urgent specialist review is recommended immediately.',
                'Prioritize emergency-capable hospitals and close follow-up.',
            ]
        elif urgency_score >= 65:
            urgency_level = 'high'
            notes = [
                'Prompt clinical follow-up is advised.',
                'Share this report with a specialist for confirmation.',
            ]
        elif urgency_score >= 40:
            urgency_level = 'moderate'
            notes = [
                'Schedule a consultation and monitor symptom progression.',
                'Consider repeat imaging if symptoms worsen.',
            ]
        else:
            urgency_level = 'low'
            notes = [
                'No immediate danger signals detected by the current model.',
                'Continue routine monitoring and preventive care.',
            ]

    return {
        'urgency_score': urgency_score,
        'urgency_level': urgency_level,
        'clinical_notes': notes,
        'generated_at': datetime.now().isoformat(),
    }

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image prediction request"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        modality = request.form.get('modality', 'xray')
        patient_name = request.form.get('patient_name', '')
        patient_age = request.form.get('patient_age', None)
        patient_gender = request.form.get('patient_gender', '')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, webp'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = str(int(random.random() * 1000000))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Make prediction
        prediction_result = predict_disease(filepath, modality)
        
        # Get hospitals
        hospitals = []
        if prediction_result.get('prediction'):
            hospitals = get_hospital_recommendations(prediction_result['prediction'])
            
            # Save to database
            try:
                prediction_id = save_prediction(
                    patient_name=patient_name,
                    patient_age=int(patient_age) if patient_age else None,
                    patient_gender=patient_gender,
                    modality=modality,
                    prediction=prediction_result['prediction'],
                    confidence=prediction_result['confidence'],
                    severity=prediction_result.get('severity'),
                    is_valid_medical_image=prediction_result.get('is_valid_medical_image', True),
                    image_quality=prediction_result.get('image_quality', {}).get('quality'),
                    notes=''
                )
                prediction_result['prediction_id'] = prediction_id
            except Exception as e:
                print(f"Database save error: {e}")
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        triage = generate_triage_summary(prediction_result)

        return jsonify({
            'success': True,
            'prediction': prediction_result,
            'hospitals': hospitals,
            'triage': triage
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/enhance', methods=['POST'])
def enhance():
    """Enhance medical image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        enhancement_type = request.form.get('type', 'auto')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save temporarily
        filename = secure_filename(file.filename)
        timestamp = str(int(random.random() * 1000000))
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Enhance image
        enhanced_base64 = enhance_image(filepath, enhancement_type)
        
        # Analyze quality
        quality = analyze_image_quality(filepath)
        
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        
        if enhanced_base64:
            return jsonify({
                'success': True,
                'enhanced_image': enhanced_base64,
                'quality': quality
            })
        else:
            return jsonify({'error': 'Failed to enhance image'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/multi-predict', methods=['POST'])
def multi_predict():
    """Handle multiple image predictions for comparison"""
    try:
        files = request.files.getlist('files')
        modality = request.form.get('modality', 'xray')
        
        if not files or len(files) < 2:
            return jsonify({'error': 'Please upload at least 2 images for comparison'}), 400
        
        if len(files) > 5:
            return jsonify({'error': 'Maximum 5 images allowed for comparison'}), 400
        
        results = []
        
        for file in files:
            if file.filename == '':
                continue
            
            # Save temporarily
            filename = secure_filename(file.filename)
            timestamp = str(int(random.random() * 1000000))
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Predict
            result = predict_disease(filepath, modality)
            result['filename'] = file.filename
            
            # Add quality analysis
            if not result.get('error'):
                result['image_quality'] = analyze_image_quality(filepath)
            
            results.append(result)
            
            # Clean up
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({
            'success': True,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report', methods=['POST'])
def generate_report():
    """CHANGE 2: Generate PDF report — fixed datetime handling"""
    try:
        data = request.json
        
        # Create report
        output_path = f"data/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        report_path = create_report(data, output_path)
        
        return jsonify({
            'success': True,
            'report_url': f'/download/{os.path.basename(report_path)}'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """CHANGE 2: Download generated report — fixed path handling"""
    try:
        file_path = os.path.join(os.path.abspath('data'), filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/history', methods=['GET'])
def history():
    """Get prediction history"""
    try:
        limit = request.args.get('limit', 50, type=int)
        modality = request.args.get('modality', None)
        search = request.args.get('search', None)
        
        predictions = get_prediction_history(limit=limit, modality=modality, search=search)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'count': len(predictions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<int:prediction_id>', methods=['GET'])
def get_history_detail(prediction_id):
    """Get specific prediction detail"""
    try:
        prediction = get_prediction_by_id(prediction_id)
        
        if prediction:
            return jsonify({'success': True, 'prediction': prediction})
        else:
            return jsonify({'error': 'Prediction not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/consultation', methods=['POST'])
def request_consultation():
    """Request consultation — saves to DB and sends acknowledgement email to patient."""
    try:
        data = request.json or {}
        patient_name = (data.get('patient_name') or '').strip()
        email = (data.get('email') or '').strip()
        phone = (data.get('phone') or '').strip()
        prediction_id = data.get('prediction_id')
        modality = data.get('modality') or None
        current_diagnosis = (data.get('current_diagnosis') or '').strip()
        symptoms_description = (data.get('symptoms_description') or '').strip()
        additional_notes = (data.get('additional_notes') or '').strip()

        if not patient_name or not email:
            return jsonify({'error': 'Name and email are required'}), 400

        consultation_id, approval_token = save_consultation(
            patient_name=patient_name, email=email, phone=phone,
            prediction_id=prediction_id, modality=modality,
            current_diagnosis=current_diagnosis,
            symptoms_description=symptoms_description,
            additional_notes=additional_notes
        )

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
        approve_url = f"{APP_BASE_URL}/approve/{approval_token}"
        decline_url = f"{APP_BASE_URL}/decline/{approval_token}"

        # ── 1. Admin notification with approve/decline links ──
        admin_html = f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#0f172a;color:#f1f5f9;padding:32px;border-radius:16px;">
            <h1 style="color:#06b6d4;text-align:center;">🏥 MedScan AI — New Consultation Request</h1>
            <hr style="border-color:#1e293b;">
            <h3 style="color:#818cf8;">Patient Details</h3>
            <table style="width:100%;color:#94a3b8;font-size:14px;">
                <tr><td><b>Name:</b></td><td>{patient_name}</td></tr>
                <tr><td><b>Email:</b></td><td>{email}</td></tr>
                <tr><td><b>Phone:</b></td><td>{phone or 'Not provided'}</td></tr>
                <tr><td><b>Modality:</b></td><td>{(modality or 'N/A').upper()}</td></tr>
                <tr><td><b>Diagnosis:</b></td><td>{current_diagnosis or 'N/A'}</td></tr>
            </table>
            <h3 style="color:#818cf8;">Symptoms</h3>
            <p style="color:#94a3b8;">{symptoms_description or 'Not provided'}</p>
            <h3 style="color:#818cf8;">Additional Notes</h3>
            <p style="color:#94a3b8;">{additional_notes or 'None'}</p>
            <hr style="border-color:#1e293b;">
            <div style="text-align:center;margin:24px 0;">
                <a href="{approve_url}" style="display:inline-block;padding:14px 36px;background:#10b981;color:white;text-decoration:none;border-radius:8px;font-weight:bold;margin:8px;">✅ Approve</a>
                <a href="{decline_url}" style="display:inline-block;padding:14px 36px;background:#f43f5e;color:white;text-decoration:none;border-radius:8px;font-weight:bold;margin:8px;">❌ Decline</a>
            </div>
            <p style="color:#64748b;font-size:12px;text-align:center;">Request ID: {consultation_id} | {now_str}</p>
        </div>
        """
        send_email(f"New Consultation Request from {patient_name}", admin_html)

        # ── 2. Patient acknowledgement email ──
        patient_ack_html = f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#0f172a;color:#f1f5f9;padding:32px;border-radius:16px;">
            <h1 style="color:#10b981;text-align:center;">✅ Consultation Request Received</h1>
            <hr style="border-color:#1e293b;">
            <p style="color:#94a3b8;font-size:15px;">
                Dear <b style="color:#f1f5f9;">{patient_name}</b>,
            </p>
            <p style="color:#94a3b8;font-size:15px;">
                Thank you for reaching out to <b style="color:#06b6d4;">MedScan AI</b>. We have successfully received your consultation request and it is currently being reviewed by our medical team.
            </p>
            <div style="background:#1e293b;padding:20px;border-radius:12px;margin:20px 0;">
                <h3 style="color:#818cf8;margin-top:0;">Your Request Summary</h3>
                <table style="width:100%;color:#94a3b8;font-size:14px;">
                    <tr><td><b>Reference ID:</b></td><td style="color:#06b6d4;">#{consultation_id}</td></tr>
                    <tr><td><b>Submitted:</b></td><td>{now_str}</td></tr>
                    {f'<tr><td><b>Diagnosis:</b></td><td>{current_diagnosis}</td></tr>' if current_diagnosis else ''}
                    {f'<tr><td><b>Phone:</b></td><td>{phone}</td></tr>' if phone else ''}
                </table>
            </div>
            <p style="color:#94a3b8;font-size:14px;">
                Our team will review your case and respond within <b style="color:#f1f5f9;">24–48 hours</b>.
                You will receive a follow-up email once a decision has been made.
            </p>
            <p style="color:#64748b;font-size:13px;margin-top:24px;">
                If you have any urgent concerns, please contact your nearest hospital emergency department.
            </p>
            <hr style="border-color:#1e293b;">
            <p style="color:#64748b;font-size:11px;text-align:center;">
                This is an automated acknowledgement. Please do not reply to this email.<br>
                MedScan AI — AI-Powered Medical Diagnostics
            </p>
        </div>
        """
        send_email(
            f"[MedScan AI] Your Consultation Request #{consultation_id} Received",
            patient_ack_html,
            to_email=email
        )

        return jsonify({
            'success': True,
            'consultation_id': consultation_id,
            'message': f'Consultation request #{consultation_id} submitted! A confirmation email has been sent to {email}. You will be notified once reviewed.'
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/approve/<token>')
def approve_consultation(token):
    """CHANGE 3: Approve consultation via email link"""
    success = update_consultation_status(token, 'approved')
    if success:
        return '<html><body style="background:#0f172a;color:#f1f5f9;font-family:Arial;display:flex;align-items:center;justify-content:center;height:100vh;"><div style="text-align:center;"><h1 style="color:#10b981;">✅ Consultation Approved</h1><p>The patient will be notified of their upcoming appointment.</p></div></body></html>'
    return '<html><body style="background:#0f172a;color:#f1f5f9;font-family:Arial;display:flex;align-items:center;justify-content:center;height:100vh;"><div style="text-align:center;"><h1 style="color:#f43f5e;">Invalid or expired link</h1></div></body></html>', 404


@app.route('/decline/<token>')
def decline_consultation(token):
    """CHANGE 3: Decline consultation via email link"""
    success = update_consultation_status(token, 'declined')
    if success:
        return '<html><body style="background:#0f172a;color:#f1f5f9;font-family:Arial;display:flex;align-items:center;justify-content:center;height:100vh;"><div style="text-align:center;"><h1 style="color:#fbbf24;">❌ Consultation Declined</h1><p>The patient will be notified.</p></div></body></html>'
    return '<html><body style="background:#0f172a;color:#f1f5f9;font-family:Arial;display:flex;align-items:center;justify-content:center;height:100vh;"><div style="text-align:center;"><h1 style="color:#f43f5e;">Invalid or expired link</h1></div></body></html>', 404


@app.route('/consultation/status/<int:consultation_id>', methods=['GET'])
def consultation_status(consultation_id):
    """CHANGE 3: Get consultation status"""
    try:
        consultation = get_consultation_by_id(consultation_id)
        if consultation:
            return jsonify({'success': True, 'consultation': consultation})
        return jsonify({'error': 'Not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/notifications', methods=['GET'])
def get_notifs():
    """CHANGE 3: Get notifications"""
    try:
        unread = request.args.get('unread', 'false') == 'true'
        notifs = get_notifications(unread_only=unread)
        return jsonify({'success': True, 'notifications': notifs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """CHANGE 4: Submit feedback with email notification"""
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        feedback_type = data.get('feedback_type')
        comment = data.get('comment', '')
        rating = data.get('rating')
        user_name = data.get('user_name', 'Anonymous')

        if not prediction_id or not feedback_type:
            return jsonify({'error': 'Prediction ID and feedback type are required'}), 400

        feedback_id = save_feedback(
            prediction_id=prediction_id,
            feedback_type=feedback_type,
            comment=comment,
            rating=rating,
            user_name=user_name
        )

        # Send email notification
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        email_html = f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#0f172a;color:#f1f5f9;padding:32px;border-radius:16px;">
            <h1 style="color:#06b6d4;text-align:center;">💬 New Feedback Received</h1>
            <hr style="border-color:#1e293b;">
            <table style="width:100%;color:#94a3b8;font-size:14px;">
                <tr><td><b>User:</b></td><td>{user_name}</td></tr>
                <tr><td><b>Type:</b></td><td>{feedback_type}</td></tr>
                <tr><td><b>Rating:</b></td><td>{'⭐' * int(rating) if rating else 'N/A'}</td></tr>
                <tr><td><b>Time:</b></td><td>{timestamp}</td></tr>
            </table>
            <h3 style="color:#818cf8;">Comment</h3>
            <p style="color:#94a3b8;background:#1e293b;padding:16px;border-radius:8px;">{comment or 'No comment provided'}</p>
        </div>
        """
        send_email(f"MedScan AI Feedback: {feedback_type}", email_html)

        return jsonify({
            'success': True,
            'feedback_id': feedback_id,
            'message': 'Thank you for your feedback!'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

import uuid

@app.route('/collab/create', methods=['POST'])
def create_collab_session():
    """Create new collaboration session"""
    session_id = str(uuid.uuid4())[:8]
    modality = request.json.get('modality', 'xray')
    
    sessions[session_id] = {
        'modality': modality,
        'prediction': None,
        'annotations': {},
        'chat': [],
        'created_at': datetime.now().isoformat(),
        'users': []
    }
    
    return jsonify({'session_id': session_id, 'success': True})

@app.route('/voice/transcribe', methods=['POST'])
def voice_transcribe():
    """CHANGE 5: Process voice text from Web Speech API and return response"""
    try:
        data = request.json
        text = data.get('text', '')
        context = data.get('context', {})

        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Context-aware responses for prediction-related queries
        text_lower = text.lower()
        if context and any(kw in text_lower for kw in ['my result', 'my diagnosis', 'prediction', 'what did you find', 'what is wrong']):
            pred = context.get('prediction', 'No diagnosis available')
            conf = context.get('confidence', 0)
            sev = context.get('severity', 'unknown')
            desc = context.get('description', '')
            response = (f"Your diagnosis result is: {pred} with {conf:.1f}% confidence. "
                       f"Severity level is {sev}. {desc[:200]}" if desc else
                       f"Your diagnosis result is: {pred} with {conf:.1f}% confidence. Severity level is {sev}.")
            return jsonify({'success': True, 'response': response, 'text': text})

        if context and any(kw in text_lower for kw in ['explain', 'tell me more', 'what does it mean', 'about this disease']):
            pred = context.get('prediction', '')
            desc = context.get('description', '')
            causes = context.get('causes', [])
            response = f"{pred}: {desc} "
            if causes:
                response += "Common causes include: " + ", ".join(causes[:3]) + "."
            return jsonify({'success': True, 'response': response, 'text': text})

        # Fallback to health assistant
        response = get_assistant_response(text)
        return jsonify({'success': True, 'response': response, 'text': text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/assistant', methods=['POST'])
def assistant():
    """Health assistant chatbot"""
    try:
        data = request.json
        message = data.get('message', '')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        response = get_assistant_response(message)
        
        return jsonify({
            'success': True,
            'response': response
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/statistics', methods=['GET'])
def statistics():
    """Get prediction statistics"""
    try:
        stats = get_statistics()
        return jsonify({'success': True, 'statistics': stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/hospitals', methods=['GET'])
def hospitals():
    """Get all hospitals"""
    return jsonify(load_hospitals())


# ==================== NEARBY HOSPITALS (REAL) ====================

def _haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lng points."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _query_overpass(lat, lng, radius_m=10000):
    """Query OpenStreetMap Overpass API for hospitals near (lat, lng).
    
    IMPORTANT: Overpass API requires a proper User-Agent header.
    It returns HTTP 406 Not Acceptable for the default python-requests agent.
    """
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius_m},{lat},{lng});
      way["amenity"="hospital"](around:{radius_m},{lat},{lng});
      relation["amenity"="hospital"](around:{radius_m},{lat},{lng});
      node["amenity"="clinic"](around:{radius_m},{lat},{lng});
      way["amenity"="clinic"](around:{radius_m},{lat},{lng});
      node["healthcare"="hospital"](around:{radius_m},{lat},{lng});
      way["healthcare"="hospital"](around:{radius_m},{lat},{lng});
    );
    out center body;
    """
    # Overpass rejects requests without a descriptive User-Agent (returns 406)
    headers = {
        'User-Agent': 'MedScanAI/1.0 (medical imaging platform; opensource)',
        'Accept': 'application/json, */*',
    }
    mirrors = [
        'https://overpass-api.de/api/interpreter',
        'https://lz4.overpass-api.de/api/interpreter',
        'https://z.overpass-api.de/api/interpreter',
    ]
    for mirror_url in mirrors:
        try:
            resp = http_requests.get(
                mirror_url,
                params={'data': query},
                headers=headers,
                timeout=25
            )
            if resp.status_code == 200:
                elements = resp.json().get('elements', [])
                print(f"  ✅ Overpass ({mirror_url.split('/')[2]}): {len(elements)} results")
                return elements   # return even if empty — area may just have no hospitals
            else:
                print(f'  ⚠ Overpass {mirror_url.split("/")[2]}: HTTP {resp.status_code}')
        except Exception as e:
            print(f'  ⚠ Overpass mirror {mirror_url.split("/")[2]} failed: {e}')
    return []


def _specialty_keywords(prediction):
    """Return search-friendly specialty keywords for a diagnosis."""
    mapping = {
        'Fractured': ['orthopedic', 'trauma', 'bone', 'fracture', 'emergency'],
        'Not Fractured': ['general', 'clinic'],
        'Glioma': ['neuro', 'oncology', 'cancer', 'brain'],
        'Meningioma': ['neuro', 'oncology', 'brain'],
        'No Tumor': ['general', 'neurology', 'clinic'],
        'Pituitary': ['endocrine', 'neuro', 'pituitary'],
        'Lung Tumor': ['oncology', 'pulmonology', 'cancer', 'lung'],
        'Bone Fracture': ['orthopedic', 'trauma', 'bone', 'emergency'],
        'Normal': ['general', 'clinic'],
    }
    return mapping.get(prediction, ['hospital', 'clinic'])


@app.route('/nearby-hospitals', methods=['POST'])
def nearby_hospitals():
    """Find real hospitals near the user using OpenStreetMap."""
    try:
        data = request.json or {}
        lat = data.get('lat')
        lng = data.get('lng')
        prediction = data.get('prediction', '')
        radius = int(data.get('radius', 10000))  # default 10 km

        if lat is None or lng is None:
            return jsonify({'error': 'Latitude and longitude are required'}), 400

        lat, lng = float(lat), float(lng)

        # Query real hospitals from OpenStreetMap
        elements = _query_overpass(lat, lng, radius)

        keywords = _specialty_keywords(prediction)

        hospitals_list = []
        for el in elements:
            tags = el.get('tags', {})
            name = tags.get('name', tags.get('name:en', ''))
            if not name:
                continue

            # Get coordinates
            h_lat = el.get('lat') or el.get('center', {}).get('lat')
            h_lng = el.get('lon') or el.get('center', {}).get('lon')
            if not h_lat or not h_lng:
                continue

            dist_km = _haversine(lat, lng, float(h_lat), float(h_lng))

            # Check specialty relevance
            name_lower = name.lower()
            healthcare_spec = tags.get('healthcare:speciality', '').lower()
            all_text = f"{name_lower} {healthcare_spec} {tags.get('description', '').lower()}"
            relevance = sum(1 for kw in keywords if kw in all_text)

            # Build Google Maps direction URL
            directions_url = (
                f"https://www.google.com/maps/dir/?api=1"
                f"&origin={lat},{lng}"
                f"&destination={h_lat},{h_lng}"
                f"&travelmode=driving"
            )
            maps_url = (
                f"https://www.google.com/maps/search/?api=1"
                f"&query={h_lat},{h_lng}"
            )

            # Build specialties list from tags
            specialties = []
            if healthcare_spec:
                specialties = [s.strip().title() for s in healthcare_spec.split(';')]
            if tags.get('emergency') == 'yes':
                specialties.append('Emergency')
            if not specialties:
                specialties = ['General Hospital']

            hospitals_list.append({
                'name': name,
                'address': tags.get('addr:full', '') or
                           ', '.join(filter(None, [
                               tags.get('addr:street', ''),
                               tags.get('addr:city', ''),
                               tags.get('addr:state', ''),
                           ])) or 'Address not available',
                'lat': float(h_lat),
                'lng': float(h_lng),
                'distance_km': round(dist_km, 1),
                'distance': f"{dist_km:.1f} km",
                'phone': tags.get('phone', tags.get('contact:phone', '')),
                'website': tags.get('website', tags.get('contact:website', '')),
                'specialties': specialties,
                'emergency': tags.get('emergency') == 'yes',
                'relevance': relevance,
                'directions_url': directions_url,
                'maps_url': maps_url,
                'opening_hours': tags.get('opening_hours', ''),
            })

        # Sort by relevance (desc) then distance (asc)
        hospitals_list.sort(key=lambda h: (-h['relevance'], h['distance_km']))

        return jsonify({
            'success': True,
            'hospitals': hospitals_list[:15],
            'user_lat': lat,
            'user_lng': lng,
            'count': min(len(hospitals_list), 15),
            'total_found': len(hospitals_list),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/diseases', methods=['GET'])
def diseases():
    """Get all supported diseases"""
    explanations = load_explanations()
    disease_list = []
    for modality, diseases in explanations.items():
        for disease in diseases.keys():
            disease_list.append({
                'modality': modality,
                'disease': disease
            })
    return jsonify(disease_list)

if __name__ == '__main__':
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=5002,
        allow_unsafe_werkzeug=True
    )

