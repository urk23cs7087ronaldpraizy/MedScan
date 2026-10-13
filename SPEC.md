# Medical Image Classification Web App - Specification

## 1. Project Overview

**Project Name:** MedScan AI - Medical Image Classification System  
**Project Type:** Full-stack Web Application (Flask + HTML/CSS/JS + TensorFlow)  
**Core Functionality:** Upload medical images (X-ray, MRI, CT) and get AI-powered disease prediction with explanations and hospital recommendations  
**Target Users:** Healthcare professionals, patients, and medical institutions

---

## 2. UI/UX Specification

### Layout Structure

**Header**
- Logo (left): "MedScan AI" with medical cross icon
- Navigation (right): Home | Upload | About | Contact
- Fixed position, height: 70px
- Semi-transparent glassmorphism effect

**Hero Section**
- Full viewport height (100vh minus header)
- Animated gradient background with floating medical icons
- Main headline and subtext
- Call-to-action button "Get Started"

**Upload Section**
- Centered card with drag-and-drop zone
- Image preview area
- Modality selector (X-ray, MRI, CT)
- Analyze button

**Results Section**
- Prediction card with confidence percentage
- Disease explanation panel
- Treatment recommendations
- Nearby hospitals map/list

**Footer**
- Copyright info
- Quick links
- Social media icons

### Visual Design

**Color Palette**
- Primary: #0EA5E9 (Medical Blue)
- Secondary: #6366F1 (Indigo)
- Accent: #10B981 (Emerald Green)
- Background: #0F172A (Dark Navy)
- Surface: #1E293B (Slate)
- Text Primary: #F8FAFC
- Text Secondary: #94A3B8
- Error: #EF4444
- Warning: #F59E0B
- Success: #22C55E

**Typography**
- Headings: "Outfit", sans-serif (Google Fonts)
- Body: "DM Sans", sans-serif (Google Fonts)
- Hero Title: 64px, bold
- Section Title: 36px, semibold
- Body Text: 16px, regular
- Small Text: 14px

**Spacing System**
- Base unit: 8px
- Section padding: 80px vertical
- Card padding: 32px
- Element gap: 16px
- Border radius: 16px (cards), 8px (buttons)

**Visual Effects**
- Glassmorphism cards (backdrop-blur: 20px)
- Gradient borders with animation
- Floating particles background
- Smooth hover transitions (0.3s ease)
- Pulsing loading animations
- Success/error toast notifications

### Components

**Navigation Bar**
- States: default, scrolled (more opaque), mobile (hamburger menu)
- Hover: underline animation

**Upload Zone**
- States: default, drag-over (highlighted border), uploading (progress), error
- Dashed border with gradient
- Icon and text centered

**Prediction Card**
- Disease name with icon
- Confidence meter (animated progress bar)
- Color-coded: green (>80%), yellow (50-80%), red (<50%)

**Explanation Panel**
- Collapsible accordion sections
- Cause, Symptoms, Risk Factors tabs

**Hospital Card**
- Hospital name and rating
- Distance from user
- Specialties badges
- Book appointment button

---

## 3. Functionality Specification

### Core Features

**1. Image Upload & Processing**
- Drag-and-drop or click to upload
- Support formats: JPEG, PNG, DICOM (converted to PNG)
- Max file size: 10MB
- Image preprocessing: resize to 224x224, normalization

**2. Modality Selection**
- X-ray
- MRI
- CT Scan

**3. Disease Classification (AI Model)**
- Trainable CNN model using TensorFlow/Keras
- Pre-trained base: ResNet50/VGG16 (transfer learning)
- Classes:
  - X-ray: Normal, Pneumonia, Tuberculosis, COVID-19
  - MRI: Normal, Brain Tumor, Alzheimer's
  - CT: Normal, Lung Tumor, Bone Fracture

**4. Disease Explanation**
- Dynamic content based on prediction
- Sections: Description, Causes, Symptoms, Risk Factors
- JSON-based explanation database

**5. Hospital Recommendations**
- Geolocation-based (if permitted)
- Fallback to default location
- Filter by specialty
- Display: name, address, rating, distance, phone

### User Interactions

1. User lands on homepage → sees hero with CTA
2. Clicks "Get Started" → scrolls to upload section
3. Selects modality (X-ray/MRI/CT)
4. Uploads image via drag-drop or file picker
5. Clicks "Analyze" → loading animation
6. Results displayed with prediction, explanation, hospitals
7. User can upload another image

### Data Handling

**Training Data (to be downloaded from Kaggle)**
- Chest X-ray Pneumonia Dataset
- Brain Tumor MRI Dataset
- Bone Fracture CT Dataset

**Model Storage**
- Saved as .h5 file
- Lazy loading on prediction

**Explanation Data**
- Stored in explanations.json

**Hospital Data**
- Mock database with major cities
- Can be extended to Google Places API

---

## 4. Technical Architecture

### File Structure
```
/medical-imaging-app
├── app.py                      # Flask backend
├── train_model.py              # Model training script
├── download_dataset.py         # Kaggle dataset downloader
├── models/
│   └── disease_model.h5        # Trained model
├── data/
│   ├── explanations.json       # Disease explanations
│   └── hospitals.json          # Hospital database
├── templates/
│   └── index.html              # Main HTML
├── static/
│   ├── css/
│   │   └── style.css           # Styles
│   └── js/
│       └── main.js             # Frontend logic
├── uploads/                    # User uploads
└── requirements.txt            # Dependencies
```

### Dependencies
- Flask
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Kaggle API

---

## 5. Acceptance Criteria

### Visual Checkpoints
- [ ] Dark theme with medical blue accents applied
- [ ] Glassmorphism cards visible
- [ ] Floating animations smooth
- [ ] Responsive on mobile/tablet/desktop

### Functional Checkpoints
- [ ] Image upload works (drag-drop and click)
- [ ] Modality selection functional
- [ ] Prediction returns result with confidence
- [ ] Disease explanation displays correctly
- [ ] Hospital suggestions show with details
- [ ] Loading states visible during processing

### Performance
- [ ] Page loads under 3 seconds
- [ ] Prediction completes under 10 seconds
- [ ] No console errors

---

## 6. Implementation Priority

1. **Phase 1:** Project setup, Flask app, HTML/CSS structure
2. **Phase 2:** Upload functionality, image preprocessing
3. **Phase 3:** Model integration, prediction logic
4. **Phase 4:** Explanation database, hospital suggestions
5. **Phase 5:** Polish, animations, responsive design

