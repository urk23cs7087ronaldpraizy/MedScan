"""
Database module for MedScan AI
CHANGE 3 & 6: Added consultation approval tokens, status updates,
and notification tracking.
"""

import sqlite3
import os
import uuid
from datetime import datetime

DB_PATH = 'data/medscan.db'


def _get_conn():
    """Get a database connection with row factory."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database with tables"""
    os.makedirs('data', exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Predictions history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            patient_age INTEGER,
            patient_gender TEXT,
            modality TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT,
            is_valid_medical_image INTEGER,
            image_quality TEXT,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Consultation requests table — CHANGE 3: added approval_token
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT,
            prediction_id INTEGER,
            modality TEXT NOT NULL,
            current_diagnosis TEXT,
            symptoms_description TEXT,
            additional_notes TEXT,
            status TEXT DEFAULT 'pending',
            approval_token TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    ''')

    # User feedback table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            feedback_type TEXT NOT NULL,
            comment TEXT,
            rating INTEGER,
            user_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    ''')

    # CHANGE 3: Notifications table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            consultation_id INTEGER,
            message TEXT NOT NULL,
            type TEXT DEFAULT 'info',
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (consultation_id) REFERENCES consultations(id)
        )
    ''')

    # Migrate existing tables — add missing columns safely
    try:
        cursor.execute("ALTER TABLE consultations ADD COLUMN approval_token TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE consultations ADD COLUMN updated_at TIMESTAMP")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE feedback ADD COLUMN user_name TEXT")
    except sqlite3.OperationalError:
        pass
    # Make modality nullable (recreate if needed is not possible in SQLite ALTER,
    # but we handle it in the INSERT instead)
    try:
        cursor.execute("UPDATE consultations SET modality = 'unknown' WHERE modality IS NULL")
    except:
        pass

    conn.commit()
    conn.close()
    print("Database initialized successfully!")


def save_prediction(patient_name=None, patient_age=None, patient_gender=None,
                    modality=None, prediction=None, confidence=None, severity=None,
                    is_valid_medical_image=None, image_quality=None, notes=None):
    """Save a prediction to the database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO predictions 
        (patient_name, patient_age, patient_gender, modality, prediction, 
         confidence, severity, is_valid_medical_image, image_quality, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_name, patient_age, patient_gender, modality, prediction,
          confidence, severity, is_valid_medical_image, image_quality, notes))

    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return prediction_id


def get_prediction_history(limit=50, modality=None, search=None):
    """Get prediction history"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM predictions WHERE 1=1"
    params = []

    if modality:
        query += " AND modality = ?"
        params.append(modality)

    if search:
        query += " AND (patient_name LIKE ? OR prediction LIKE ?)"
        params.extend([f'%{search}%', f'%{search}%'])

    query += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_prediction_by_id(prediction_id):
    """Get a specific prediction by ID"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
    row = cursor.fetchone()
    conn.close()

    return dict(row) if row else None


def save_consultation(patient_name, email, phone=None, prediction_id=None,
                     modality=None, current_diagnosis=None, symptoms_description=None,
                     additional_notes=None):
    """CHANGE 3: Save consultation with unique approval token"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Generate unique approval token
    approval_token = str(uuid.uuid4())

    # modality may be None — default to 'unknown' to satisfy old NOT NULL constraint
    modality_val = modality or 'unknown'

    cursor.execute('''
        INSERT INTO consultations
        (patient_name, email, phone, prediction_id, modality, 
         current_diagnosis, symptoms_description, additional_notes, approval_token)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (patient_name, email, phone, prediction_id, modality_val,
          current_diagnosis, symptoms_description, additional_notes, approval_token))

    consultation_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return consultation_id, approval_token


def get_consultations(status=None):
    """Get consultation requests"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if status:
        cursor.execute("SELECT * FROM consultations WHERE status = ? ORDER BY created_at DESC", (status,))
    else:
        cursor.execute("SELECT * FROM consultations ORDER BY created_at DESC")

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows]


def get_consultation_by_id(consultation_id):
    """CHANGE 3: Get consultation by ID"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM consultations WHERE id = ?", (consultation_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_consultation_by_token(token):
    """CHANGE 3: Get consultation by approval token"""
    conn = _get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM consultations WHERE approval_token = ?", (token,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def update_consultation_status(token, status):
    """CHANGE 3: Update consultation status via approval token"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        UPDATE consultations SET status = ?, updated_at = ?
        WHERE approval_token = ?
    ''', (status, datetime.now().isoformat(), token))

    affected = cursor.rowcount
    conn.commit()

    # Create notification if approved
    if affected > 0:
        cursor2 = conn.cursor()
        cursor2.execute("SELECT id, patient_name FROM consultations WHERE approval_token = ?", (token,))
        row = cursor2.fetchone()
        if row:
            consultation_id = row[0]
            patient_name = row[1]
            if status == 'approved':
                msg = f"Your consultation request has been approved! An appointment will be scheduled for {patient_name}."
                notif_type = 'success'
            else:
                msg = f"Consultation request for {patient_name} has been declined."
                notif_type = 'warning'

            cursor2.execute('''
                INSERT INTO notifications (consultation_id, message, type)
                VALUES (?, ?, ?)
            ''', (consultation_id, msg, notif_type))

    conn.commit()
    conn.close()

    return affected > 0


def get_notifications(limit=20, unread_only=False):
    """CHANGE 3: Get notifications"""
    conn = _get_conn()
    cursor = conn.cursor()

    if unread_only:
        cursor.execute("SELECT * FROM notifications WHERE is_read = 0 ORDER BY created_at DESC LIMIT ?", (limit,))
    else:
        cursor.execute("SELECT * FROM notifications ORDER BY created_at DESC LIMIT ?", (limit,))

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def mark_notification_read(notification_id):
    """CHANGE 3: Mark notification as read"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE notifications SET is_read = 1 WHERE id = ?", (notification_id,))
    conn.commit()
    conn.close()


def save_feedback(prediction_id, feedback_type, comment=None, rating=None, user_name=None):
    """Save user feedback"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        INSERT INTO feedback (prediction_id, feedback_type, comment, rating, user_name)
        VALUES (?, ?, ?, ?, ?)
    ''', (prediction_id, feedback_type, comment, rating, user_name))

    feedback_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return feedback_id


def get_statistics():
    """Get prediction statistics"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    # By modality
    cursor.execute("SELECT modality, COUNT(*) FROM predictions GROUP BY modality")
    by_modality = dict(cursor.fetchall())

    # By prediction
    cursor.execute("SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction")
    by_prediction = dict(cursor.fetchall())

    # Average confidence
    cursor.execute("SELECT AVG(confidence) FROM predictions WHERE confidence > 0")
    avg_confidence = cursor.fetchone()[0] or 0

    # Valid vs invalid images
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN is_valid_medical_image = 1 THEN 1 ELSE 0 END) as valid,
            SUM(CASE WHEN is_valid_medical_image = 0 THEN 1 ELSE 0 END) as invalid
        FROM predictions
    """)
    valid_stats = cursor.fetchone()

    conn.close()

    return {
        'total': total,
        'by_modality': by_modality,
        'by_prediction': by_prediction,
        'avg_confidence': round(avg_confidence, 2),
        'valid_images': valid_stats[0] or 0,
        'invalid_images': valid_stats[1] or 0
    }


# Initialize database on import
init_db()
