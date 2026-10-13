"""
PDF Report Generator for MedScan AI
Generates professional medical reports
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
import os


def create_report(data, output_path='data/report.pdf'):
    """
    Create a PDF medical report
    
    Args:
        data: Dictionary containing:
            - patient_name: str
            - patient_age: int
            - patient_gender: str
            - modality: str (xray/mri/ct)
            - prediction: str
            - confidence: float
            - severity: str
            - description: str
            - causes: list
            - symptoms: list
            - risk_factors: list
            - treatment: list
            - prevention: list
            - hospitals: list
            - created_at: datetime
    
    Returns:
        Path to generated PDF
    """
    os.makedirs('data', exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        textColor=colors.HexColor('#0EA5E9'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=10,
        spaceBefore=20
    )
    
    subheading_style = ParagraphStyle(
        'SubHeading',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor('#6366F1'),
        spaceAfter=8
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        leading=14
    )
    
    # Build the story
    story = []
    
    # Header
    story.append(Paragraph("MedScan AI", title_style))
    story.append(Paragraph("Medical Image Analysis Report", title_style))
    story.append(Spacer(1, 20))
    
    # Report metadata
    # CHANGE 2 FIX: Handle created_at as string or datetime, fix syntax error
    created_at = data.get('created_at', datetime.now())
    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            created_at = datetime.now()
    report_date = created_at.strftime('%B %d, %Y at %H:%M')
    report_id = f"MED-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    story.append(Paragraph(f"<b>Report Date:</b> {report_date}", normal_style))
    story.append(Paragraph(f"<b>Report ID:</b> {report_id}", normal_style))
    story.append(Spacer(1, 20))
    
    # Patient Information Section
    story.append(Paragraph("Patient Information", heading_style))
    
    patient_data = [
        ['Name:', data.get('patient_name', 'Not provided')],
        ['Age:', str(data.get('patient_age', 'Not provided'))],
        ['Gender:', data.get('patient_gender', 'Not provided')],
        ['Modality:', data.get('modality', '').upper()],
    ]
    
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 20))
    
    # Diagnosis Section
    story.append(Paragraph("Diagnosis Results", heading_style))
    
    # Color based on severity
    severity_colors = {
        'Normal': '#22C55E',
        'Mild': '#F59E0B',
        'Moderate': '#EF4444',
        'Severe': '#DC2626'
    }
    severity = data.get('severity', 'Normal')
    severity_color = severity_colors.get(severity, '#6366F1')
    
    diagnosis_data = [
        ['Primary Diagnosis:', data.get('prediction', 'N/A')],
        ['Confidence Level:', f"{data.get('confidence', 0):.1f}%"],
        ['Severity:', f"<font color='{severity_color}'><b>{severity}</b></font>"],
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(diagnosis_table)
    story.append(Spacer(1, 20))
    
    # Differential Diagnoses if available
    if data.get('differential_diagnoses'):
        story.append(Paragraph("Differential Diagnoses", heading_style))
        
        diff_data = []
        for diag in data['differential_diagnoses']:
            diff_data.append([diag['prediction'], f"{diag['confidence']:.1f}%"])
        
        diff_table = Table(diff_data, colWidths=[4*inch, 1*inch])
        diff_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        story.append(diff_table)
        story.append(Spacer(1, 20))
    
    # Disease Description
    if data.get('description'):
        story.append(Paragraph("Disease Description", heading_style))
        story.append(Paragraph(data['description'], normal_style))
        story.append(Spacer(1, 15))
    
    # Causes
    if data.get('causes'):
        story.append(Paragraph("Causes", subheading_style))
        for cause in data['causes']:
            story.append(Paragraph(f"• {cause}", normal_style))
        story.append(Spacer(1, 15))
    
    # Symptoms
    if data.get('symptoms'):
        story.append(Paragraph("Symptoms", subheading_style))
        for symptom in data['symptoms']:
            story.append(Paragraph(f"• {symptom}", normal_style))
        story.append(Spacer(1, 15))
    
    # Risk Factors
    if data.get('risk_factors'):
        story.append(Paragraph("Risk Factors", subheading_style))
        for risk in data['risk_factors']:
            story.append(Paragraph(f"• {risk}", normal_style))
        story.append(Spacer(1, 15))
    
    # Treatment
    if data.get('treatment'):
        story.append(Paragraph("Treatment Options", subheading_style))
        for t in data['treatment']:
            story.append(Paragraph(f"• {t}", normal_style))
        story.append(Spacer(1, 15))
    
    # Prevention
    if data.get('prevention'):
        story.append(Paragraph("Prevention", subheading_style))
        for p in data['prevention']:
            story.append(Paragraph(f"• {p}", normal_style))
        story.append(Spacer(1, 20))
    
    # Hospital Recommendations
    if data.get('hospitals'):
        story.append(Paragraph("Recommended Healthcare Facilities", heading_style))
        
        for i, hospital in enumerate(data['hospitals'][:3], 1):
            story.append(Paragraph(f"<b>{i}. {hospital.get('name', 'N/A')}</b>", normal_style))
            story.append(Paragraph(f"   Address: {hospital.get('address', 'N/A')}", normal_style))
            if hospital.get('phone'):
                story.append(Paragraph(f"   Phone: {hospital.get('phone', '')}", normal_style))
            if hospital.get('rating'):
                story.append(Paragraph(f"   Rating: {'★' * int(hospital.get('rating', 0))}", normal_style))
            story.append(Spacer(1, 10))
    
    # Disclaimer
    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>Disclaimer</b>", subheading_style))
    disclaimer_text = (
        "This report is generated by MedScan AI for informational purposes only. "
        "It is not intended to replace professional medical advice, diagnosis, or treatment. "
        "Always consult with a qualified healthcare provider for medical decisions. "
        "The accuracy of this analysis depends on the quality of the uploaded medical images "
        "and may not be 100% accurate."
    )
    story.append(Paragraph(disclaimer_text, normal_style))
    
    # Footer
    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Generated by MedScan AI | www.medscanai.com | support@medscanai.com",
        ParagraphStyle('Footer', parent=normal_style, fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
    ))
    
    # Build PDF
    doc.build(story)
    
    return output_path


def generate_quick_report(prediction, confidence, modality):
    """Generate a quick summary report"""
    data = {
        'prediction': prediction,
        'confidence': confidence,
        'modality': modality,
        'severity': 'Normal' if prediction == 'Normal' else 'Requires Review',
        'created_at': datetime.now()
    }
    
    output_path = f"data/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    return create_report(data, output_path)

