# MedScan AI

MedScan AI is a Flask-based medical image analysis web application for classifying diagnostic images such as X-rays, MRI scans, and CT images. It combines AI-powered prediction, image enhancement, medical explanations, hospital recommendations, consultation workflows, and a health assistant to support clinicians and patients.

## Overview

This project allows users to:

- Upload medical images for analysis
- Select a modality such as X-ray, MRI, or CT
- Get disease predictions with confidence scores
- Review image quality and enhancement options
- View explanation text and triage guidance
- Find nearby hospitals based on predicted conditions
- Save and review prediction history
- Submit consultation requests and feedback
- Collaborate in real time and interact with a health assistant

## Features

- AI-based medical image classification
- Fallback prediction support when model dependencies are unavailable
- Image validation and quality analysis
- Image enhancement tools
- PDF report generation
- Prediction history and statistics
- Email-based consultation workflow
- Nearby hospital lookup using geographic data
- Voice/text health assistant interaction
- Real-time collaboration session support

## Tech Stack

- Python 3
- Flask
- Flask-SocketIO
- OpenCV
- NumPy
- Pillow
- scikit-learn / joblib
- TensorFlow / Keras (optional fallback)
- SQLAlchemy
- ReportLab
- JavaScript + HTML + CSS

## Project Structure

```text
.
├── app.py                     # Main Flask application and API routes
├── database.py               # Database logic for predictions, consultations, notifications
├── collab.py                 # Real-time collaboration utilities
├── download_data.py          # Dataset download utilities
├── health_assistant.py       # Health assistant logic
├── image_enhancer.py         # Image enhancement and severity analysis
├── report_generator.py       # PDF generation
├── train_classifier.py       # Model training helper
├── train_dataset.py          # Dataset preparation tooling
├── train_model.py            # Training pipeline
├── validate_image.py         # Image validation and quality checks
├── voice_assistant.py        # Voice-related workflows
├── models/                   # Trained model files and class metadata
├── data/                     # Explanations, hospitals, and generated reports
├── dataset/                  # Medical image datasets
├── static/                   # CSS and JavaScript frontend assets
├── templates/                # HTML templates
├── uploads/                  # Temporary uploaded files
├── requirements.txt          # Python dependencies
├── SPEC.md                  # Product/specification notes
├── TODO.md                  # Project task list
├── README.md                 # Project documentation
└── .env.example or project config files (if added later)
```

## Installation

1. Clone the project repository.
2. Navigate to the project directory.
3. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Flask app with:

```bash
python3 app.py
```

The app typically runs at:

```text
http://127.0.0.1:5002
```

If you are using a different environment, ensure the app is started with Python 3, not the plain `python` alias if it is not configured on your machine.

## Environment Variables

Some features such as consultation and notification emails rely on environment variables.

```bash
export MAIL_USERNAME="your-gmail-address"
export MAIL_PASSWORD="your-app-password"
export APP_BASE_URL="http://localhost:5002"
```

If these are not set, the app still runs, but email delivery may be skipped gracefully.

## App Workflow

1. Open the homepage in the browser.
2. Choose the medical image modality.
3. Upload a valid image.
4. Submit the prediction request.
5. Review prediction, confidence, explanation, and recommended next steps.
6. Optionally generate a report, request a consultation, or use the health assistant.

## API Endpoints

The app exposes several endpoints for prediction and supporting features, including:

- `GET /` — homepage
- `POST /predict` — submit image and get diagnosis
- `POST /enhance` — enhance uploaded image
- `POST /multi-predict` — compare several images
- `POST /report` — generate a PDF report
- `GET /history` — view prediction history
- `POST /consultation` — submit consultation request
- `POST /feedback` — save user feedback
- `POST /assistant` — chat with health assistant
- `GET /statistics` — fetch summary statistics
- `GET /hospitals` — retrieve hospital list
- `POST /nearby-hospitals` — get local hospital recommendations

## Model Notes

The app attempts to load trained sklearn model files from the `models/` directory first. If not available, it can fall back to TensorFlow or operate in a reduced prediction mode depending on the environment.

Supported disease categories in the current project include:

- X-ray: Fractured / Not Fractured
- MRI: Glioma / Meningioma / No Tumor / Pituitary

## Data Notes

- `data/explanations.json` stores disease explanations and supporting text.
- `data/hospitals.json` contains static hospital records.
- Uploaded files are stored temporarily in the `uploads/` directory.
- Generated PDFs are saved in the `data/` directory.

## Troubleshooting

### `python: command not found`

Use:

```bash
python3 app.py
```

### Missing dependencies

Reinstall the environment:

```bash
pip install -r requirements.txt
```

### TensorFlow issues

This application includes a safeguard that prevents TensorFlow from crashing the app on unsupported systems. If TensorFlow is unavailable, the app will still run in fallback mode when possible.

### Port already in use

If port `5002` is already occupied, update the application port or stop the conflicting process.

## Future Improvements

- Add more medical imaging classes
- Integrate a production-grade cloud database
- Use a more robust model deployment pipeline
- Add real hospital API integration
- Improve medical validation and clinician review workflows
- Add authentication and role-based access for institutions

## License

This project is for educational and prototyping purposes unless otherwise specified by the project owner.

## Contributing

Pull requests and improvements are welcome. If you are extending the project, keep the app compatible with existing workflows and document any changes to the model, routes, or environment variables.

## Contact

For project questions or support, refer to the repository owner or project maintainer.
