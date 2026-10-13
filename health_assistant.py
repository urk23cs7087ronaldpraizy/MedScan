"""
AI Health Assistant for MedScan AI
CHANGE 7: Enhanced NLP with fuzzy matching, disease explanations,
symptom-disease mapping, and structured responses.
"""

import json
import re
import random
from datetime import datetime

# ─── Extended Health Knowledge Base ───
HEALTH_KNOWLEDGE = {
    "general": {
        "symptoms": {
            "cough": "A cough can be caused by infections, allergies, asthma, or acid reflux. Persistent cough lasting more than 3 weeks should be evaluated by a doctor.",
            "fever": "Fever is often a sign of infection. A temperature above 100.4°F (38°C) is considered a fever. Rest, hydration, and fever-reducing medication can help. Seek medical attention if fever exceeds 103°F or persists for more than 3 days.",
            "headache": "Headaches can be caused by stress, dehydration, lack of sleep, sinus issues, or migraines. See a doctor if headaches are severe, frequent, or accompanied by vision changes.",
            "fatigue": "Fatigue can result from poor sleep, stress, poor diet, anemia, thyroid issues, or chronic conditions. Consult a doctor if fatigue persists despite adequate rest.",
            "chest_pain": "Chest pain can indicate heart attack but can also be caused by anxiety, muscle strain, or acid reflux. If severe or accompanied by shortness of breath, seek emergency care immediately.",
            "shortness_breath": "Shortness of breath can be caused by asthma, anxiety, heart conditions, lung disease, or physical exertion. If sudden or severe, seek immediate medical attention.",
            "nausea": "Nausea can be caused by food poisoning, migraine, pregnancy, anxiety, or stomach issues. Seek medical care if persistent or accompanied by severe abdominal pain.",
            "dizziness": "Dizziness can be caused by low blood sugar, dehydration, inner ear issues, or medication side effects. See a doctor if accompanied by chest pain, confusion, or fainting.",
            "back_pain": "Back pain is commonly caused by muscle strain, poor posture, herniated discs, or arthritis. Rest, ice/heat therapy, and stretching can help. See a doctor if pain radiates down legs or is accompanied by numbness.",
            "joint_pain": "Joint pain may result from arthritis, injury, gout, or infection. Rest the affected joint, apply ice, and consider anti-inflammatory medication. See a doctor for persistent or worsening pain.",
            "stomach_pain": "Stomach pain can indicate gastritis, ulcers, food poisoning, IBS, or appendicitis. Seek emergency care for severe, sudden pain especially in the lower right abdomen.",
            "sore_throat": "Sore throat is usually caused by viral infections but can also indicate strep throat. Gargle warm salt water, stay hydrated. See a doctor if it lasts more than a week.",
            "insomnia": "Insomnia can be caused by stress, anxiety, poor sleep habits, or medical conditions. Maintain a regular sleep schedule, limit caffeine, and create a restful environment.",
            "anxiety": "Anxiety is a normal response to stress but can become a disorder. Deep breathing, exercise, and mindfulness can help. Consult a mental health professional if it interferes with daily life.",
            "depression": "Depression involves persistent sadness, loss of interest, and fatigue. Exercise, social support, and therapy are helpful. Seek professional help — you are not alone."
        },
        "prevention": {
            "flu": "Get annual flu vaccine, wash hands frequently, avoid close contact with sick people, and maintain healthy lifestyle.",
            "cold": "Wash hands often, avoid touching your face, stay hydrated, get adequate sleep, and manage stress.",
            "heart_disease": "Exercise regularly, eat a balanced diet low in saturated fats, maintain healthy weight, don't smoke, limit alcohol, and manage stress.",
            "diabetes": "Maintain healthy weight, exercise regularly, eat a balanced diet, limit sugar and refined carbs, and have regular check-ups.",
            "cancer": "Don't smoke, eat a healthy diet, exercise regularly, protect from sun, limit alcohol, and get recommended screenings.",
            "stroke": "Control blood pressure, maintain healthy weight, exercise regularly, don't smoke, limit alcohol, and manage diabetes.",
            "fracture": "Ensure adequate calcium and vitamin D intake, do weight-bearing exercises, prevent falls, and wear protective gear during sports.",
            "tuberculosis": "Get BCG vaccination, avoid close contact with TB patients, maintain good ventilation, and get regular TB screening if at risk."
        },
        "lifestyle": {
            "exercise": "Adults should get at least 150 minutes of moderate aerobic activity or 75 minutes of vigorous activity per week.",
            "sleep": "Adults need 7-9 hours of sleep per night. Poor sleep can affect mood, concentration, and health.",
            "diet": "A healthy diet includes fruits, vegetables, whole grains, lean proteins, and limited processed foods, sugars, and saturated fats.",
            "water": "Drink at least 8 glasses of water daily. More if exercising or in hot weather."
        }
    },
    "modality_info": {
        "xray": {
            "name": "X-Ray",
            "description": "X-rays use radiation to create images of bones and certain tissues. They're commonly used to detect fractures, pneumonia, lung infections, and tumors.",
            "preparation": "Remove jewelry and metal objects. You may need to wear a gown. Inform your doctor if you're pregnant.",
            "risks": "X-rays involve minimal radiation exposure. The benefits usually outweigh the risks. Pregnant women should avoid x-rays unless absolutely necessary."
        },
        "mri": {
            "name": "MRI Scan",
            "description": "MRI uses magnetic fields and radio waves to create detailed images of organs and tissues. It's excellent for brain, spine, joint, and soft tissue imaging.",
            "preparation": "Remove all metal objects. Inform about metal implants, pacemakers, or tattoos. You may need to avoid caffeine before the scan.",
            "risks": "MRI is generally safe. However, people with certain metal implants cannot have MRI. The procedure is loud but painless."
        },
        "ct": {
            "name": "CT Scan",
            "description": "CT scans use X-rays to create cross-sectional images of the body. They're faster than MRI and excellent for detecting bone injuries, internal bleeding, and tumors.",
            "preparation": "You may need to fast before the scan. Remove metal objects. You may be given a contrast dye.",
            "risks": "CT involves more radiation than regular X-rays. Contrast dye can cause allergic reactions in rare cases. Drink plenty of water after to flush out the dye."
        }
    }
}

# ─── Disease Explanation Database ───
DISEASE_EXPLANATIONS = {
    "glioma": {
        "name": "Glioma",
        "what": "A glioma is a type of brain tumor that originates from glial cells, which support and nourish neurons in the brain.",
        "types": "Gliomas range from low-grade (slow-growing) to high-grade (aggressive). Glioblastoma (GBM) is the most aggressive form.",
        "symptoms": "Persistent headaches, seizures, personality changes, vision problems, nausea, and progressive neurological deficits.",
        "treatment": "Surgical resection, radiation therapy, chemotherapy (temozolomide), and targeted/immunotherapy for advanced cases.",
        "prognosis": "Depends on grade. Low-grade gliomas may be managed for years. High-grade gliomas require aggressive treatment."
    },
    "meningioma": {
        "name": "Meningioma",
        "what": "A meningioma is a tumor arising from the meninges — the protective membranes surrounding the brain and spinal cord.",
        "types": "Most (90%) are benign (WHO Grade I). Some can be atypical (Grade II) or malignant (Grade III).",
        "symptoms": "Gradual headaches, vision changes, hearing loss, memory problems, and weakness in limbs.",
        "treatment": "Observation for small tumors, surgical excision, and stereotactic radiosurgery (Gamma Knife) for symptomatic cases.",
        "prognosis": "Generally good for benign meningiomas. Most can be cured with complete surgical removal."
    },
    "pituitary": {
        "name": "Pituitary Tumor (Adenoma)",
        "what": "A growth in the pituitary gland at the base of the brain that controls hormone production.",
        "types": "Can be functional (hormone-secreting) or non-functional. Prolactinomas are the most common type.",
        "symptoms": "Vision changes, hormonal imbalances, headaches, fatigue, weight changes, and reproductive issues.",
        "treatment": "Transsphenoidal surgery, medication (dopamine agonists), radiation therapy, and hormone replacement.",
        "prognosis": "Generally excellent. Most pituitary tumors are benign and treatable."
    },
    "fracture": {
        "name": "Bone Fracture",
        "what": "A complete or partial break in the bone, often caused by trauma, overuse, or conditions that weaken bones like osteoporosis.",
        "types": "Simple (clean break), compound (bone pierces skin), stress (hairline), and comminuted (shattered).",
        "symptoms": "Acute pain, swelling, bruising, deformity, difficulty moving the affected area, and tenderness.",
        "treatment": "Immobilization (cast/splint), surgical fixation (plates, screws), pain management, and physical therapy.",
        "prognosis": "Most fractures heal in 6-12 weeks. Recovery depends on location, severity, age, and overall health."
    },
    "pneumonia": {
        "name": "Pneumonia",
        "what": "An infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus.",
        "types": "Bacterial, viral, fungal, and aspiration pneumonia. Community-acquired vs hospital-acquired.",
        "symptoms": "Cough with phlegm, fever, chills, difficulty breathing, chest pain, fatigue, and confusion in elderly.",
        "treatment": "Antibiotics (bacterial), antivirals, rest, fluids, and oxygen therapy for severe cases.",
        "prognosis": "Most people recover in 1-3 weeks. Can be serious for elderly, infants, and immunocompromised."
    },
    "tuberculosis": {
        "name": "Tuberculosis (TB)",
        "what": "A bacterial infection caused by Mycobacterium tuberculosis that primarily affects the lungs.",
        "types": "Latent TB (inactive, not contagious) and Active TB (symptomatic, contagious).",
        "symptoms": "Persistent cough (3+ weeks), coughing blood, chest pain, weight loss, night sweats, and fever.",
        "treatment": "6-9 month course of antibiotics (isoniazid, rifampicin, pyrazinamide, ethambutol).",
        "prognosis": "Curable with proper treatment. Drug-resistant TB requires longer, more complex treatment."
    },
    "lung tumor": {
        "name": "Lung Tumor/Cancer",
        "what": "An abnormal growth of cells in the lungs that can be benign or malignant (cancerous).",
        "types": "Non-small cell lung cancer (NSCLC) — most common, and small cell lung cancer (SCLC) — more aggressive.",
        "symptoms": "Persistent cough, coughing blood, chest pain, weight loss, shortness of breath, and wheezing.",
        "treatment": "Surgery, chemotherapy, radiation, targeted therapy (EGFR/ALK inhibitors), and immunotherapy.",
        "prognosis": "Depends on stage at diagnosis. Early-stage has much better outcomes. Screening saves lives."
    },
    "brain tumor": {
        "name": "Brain Tumor",
        "what": "An abnormal growth of cells in the brain. Can be primary (originating in brain) or secondary (metastatic).",
        "types": "Gliomas, meningiomas, pituitary adenomas, schwannomas, medulloblastomas, and metastatic tumors.",
        "symptoms": "Headaches, seizures, cognitive changes, vision/hearing problems, balance issues, and personality changes.",
        "treatment": "Surgery, radiation therapy, chemotherapy, targeted therapy, and supportive care.",
        "prognosis": "Varies widely based on tumor type, grade, location, and patient factors."
    }
}

# ─── Symptom Checker Patterns ───
SYMPTOM_PATTERNS = {
    "respiratory": {
        "keywords": ["cough", "breathing", "shortness", "wheezing", "chest", "lung", "breath", "asthma", "bronchitis", "phlegm", "sputum"],
        "conditions": ["Asthma", "Pneumonia", "Bronchitis", "COPD", "COVID-19", "Tuberculosis", "Lung Cancer"],
        "advice": "If you're experiencing respiratory symptoms, especially difficulty breathing, seek medical attention promptly. Rest, stay hydrated, and avoid irritants."
    },
    "cardiovascular": {
        "keywords": ["heart", "chest pain", "palpitation", "blood pressure", "pulse", "cardiac", "arrhythmia", "angina"],
        "conditions": ["Heart Disease", "Hypertension", "Arrhythmia", "Heart Attack", "Heart Failure"],
        "advice": "Chest pain or heart-related symptoms require immediate medical attention. Call emergency services if experiencing severe chest pain with arm/jaw pain, sweating, or nausea."
    },
    "neurological": {
        "keywords": ["headache", "dizziness", "numbness", "tingling", "seizure", "confusion", "memory", "migraine", "vertigo", "brain"],
        "conditions": ["Migraine", "Stroke", "Alzheimer's", "Brain Tumor", "Epilepsy", "Meningitis"],
        "advice": "Sudden severe headache, weakness, or confusion could indicate stroke. Seek emergency care. Memory issues should be evaluated by a doctor."
    },
    "digestive": {
        "keywords": ["nausea", "vomiting", "diarrhea", "constipation", "stomach", "abdomen", "appetite", "bloating", "indigestion", "acid reflux"],
        "conditions": ["Gastritis", "Food Poisoning", "IBS", "Ulcer", "Gallbladder Disease", "GERD"],
        "advice": "Stay hydrated with clear fluids. Avoid solid food until nausea passes. Seek medical care for severe abdominal pain, blood in stool, or dehydration."
    },
    "musculoskeletal": {
        "keywords": ["bone", "fracture", "joint", "muscle", "back pain", "sprain", "arthritis", "swelling", "stiffness", "osteoporosis"],
        "conditions": ["Fracture", "Arthritis", "Osteoporosis", "Tendinitis", "Muscle Strain", "Herniated Disc"],
        "advice": "Rest the affected area, apply ice for swelling, and take anti-inflammatory medication. Seek medical care for severe pain, deformity, or inability to move."
    },
    "skin": {
        "keywords": ["rash", "itching", "skin", "hives", "eczema", "acne", "mole", "burn", "wound"],
        "conditions": ["Eczema", "Psoriasis", "Dermatitis", "Skin Cancer", "Allergic Reaction"],
        "advice": "Keep the affected area clean and moisturized. Avoid scratching. See a dermatologist for persistent or worsening skin conditions."
    }
}


def _fuzzy_match(text, keywords):
    """CHANGE 7: Fuzzy keyword matching — handles partial words, plurals, and typos."""
    text_lower = text.lower()
    for kw in keywords:
        # Exact match
        if kw in text_lower:
            return True
        # Partial/stem match (at least 4 chars)
        if len(kw) >= 4 and kw[:4] in text_lower:
            return True
        # Check individual words
        for word in text_lower.split():
            if len(word) >= 3 and len(kw) >= 3:
                # Simple Levenshtein-like: if words share 70%+ characters
                common = sum(1 for c in word if c in kw)
                if common >= max(len(word), len(kw)) * 0.65:
                    return True
    return False


class HealthAssistant:
    """AI Health Assistant with enhanced NLP for conversational health advice"""

    def __init__(self):
        self.conversation_history = []
        self.max_history = 10

    def process_message(self, user_message):
        """Process user message and return appropriate response"""
        user_message_original = user_message.strip()
        user_message = user_message.lower().strip()

        self.conversation_history.append({
            "role": "user",
            "message": user_message_original,
            "timestamp": datetime.now().isoformat()
        })

        response = self.get_response(user_message)

        self.conversation_history.append({
            "role": "assistant",
            "message": response,
            "timestamp": datetime.now().isoformat()
        })

        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

        return response

    def get_response(self, message):
        """Generate response using multi-layer intent detection"""

        # Layer 1: Greetings
        if any(word in message for word in ["hello", "hi", "hey", "greetings", "good morning", "good evening"]):
            return self._greeting()

        # Layer 2: Thanks / Goodbye
        if any(word in message for word in ["thank", "thanks", "appreciate"]):
            return "You're welcome! Is there anything else I can help you with? 😊"
        if any(word in message for word in ["bye", "goodbye", "see you", "take care"]):
            return "Goodbye! Take care of your health. Feel free to come back if you have more questions! 👋"

        # Layer 3: Disease explanation queries ("what is glioma", "explain fracture", "tell me about pneumonia")
        disease_response = self._check_disease_query(message)
        if disease_response:
            return disease_response

        # Layer 4: Modality info ("xray", "mri", "ct scan")
        modality_response = self._check_modality_query(message)
        if modality_response:
            return modality_response

        # Layer 5: Symptom pattern matching (fuzzy)
        symptom_response = self._check_symptom_patterns(message)
        if symptom_response:
            return symptom_response

        # Layer 6: Specific symptom lookup
        specific_response = self._check_specific_symptoms(message)
        if specific_response:
            return specific_response

        # Layer 7: Prevention queries
        if any(word in message for word in ["prevent", "prevention", "avoid", "protect", "reduce risk"]):
            return self._get_prevention_advice(message)

        # Layer 8: Lifestyle queries
        lifestyle_response = self._check_lifestyle(message)
        if lifestyle_response:
            return lifestyle_response

        # Layer 9: Result-related queries
        if any(phrase in message for phrase in ["my result", "my diagnosis", "my prediction", "my report", "what did it say"]):
            return "To see your diagnosis result, please go to the **Analyze** section, upload a medical image, and click **Analyze Image**. Your results will appear with detailed explanations, confidence scores, and hospital recommendations.\n\nIf you've already analyzed an image, scroll up to the **Results** section to review your diagnosis."

        # Layer 10: Help / capabilities
        if any(word in message for word in ["help", "what can you do", "capabilities", "features"]):
            return self._greeting()

        # Fallback
        return self._fallback_response()

    def _greeting(self):
        return ("Hello! I'm your MedScan AI Health Assistant. I can help you with:\n\n"
                "• **Disease Information** — Ask \"What is glioma?\" or \"Explain pneumonia\"\n"
                "• **Symptom Analysis** — Describe your symptoms and I'll suggest possible conditions\n"
                "• **Medical Scans** — Learn about X-ray, MRI, and CT scans\n"
                "• **Prevention Tips** — Ask \"How to prevent diabetes?\"\n"
                "• **Lifestyle Advice** — Questions about exercise, diet, sleep, hydration\n\n"
                "How can I help you today?")

    def _check_disease_query(self, message):
        """CHANGE 7: Check if user is asking about a specific disease"""
        trigger_phrases = ["what is", "what's", "explain", "tell me about", "describe", "information about",
                           "info on", "details about", "what are", "know about", "learn about", "meaning of"]

        is_asking = any(phrase in message for phrase in trigger_phrases)

        for disease_key, info in DISEASE_EXPLANATIONS.items():
            if disease_key in message or info["name"].lower() in message:
                if is_asking or disease_key in message:
                    return self._format_disease_explanation(info)

        return None

    def _format_disease_explanation(self, info):
        """Format a structured disease explanation"""
        response = f"📋 **{info['name']}**\n\n"
        response += f"**What is it?**\n{info['what']}\n\n"
        if info.get('types'):
            response += f"**Types:** {info['types']}\n\n"
        response += f"**Common Symptoms:**\n{info['symptoms']}\n\n"
        response += f"**Treatment:**\n{info['treatment']}\n\n"
        response += f"**Prognosis:** {info['prognosis']}\n\n"
        response += "⚠️ This is general information only. Always consult a healthcare professional for proper diagnosis and treatment."
        return response

    def _check_modality_query(self, message):
        """Check for imaging modality questions"""
        modality_map = {
            "xray": ["xray", "x-ray", "x ray", "radiograph"],
            "mri": ["mri", "magnetic resonance", "brain scan"],
            "ct": ["ct scan", "cat scan", "computed tomography", "ct imaging"]
        }

        for modality, keywords in modality_map.items():
            if any(kw in message for kw in keywords):
                return self._get_modality_info(modality)

        return None

    def _get_modality_info(self, modality):
        """Get formatted modality information"""
        info = HEALTH_KNOWLEDGE["modality_info"].get(modality)
        if not info:
            return None

        response = f"📷 **{info['name']}**\n\n"
        response += f"**Description:** {info['description']}\n\n"
        response += f"**Preparation:** {info['preparation']}\n\n"
        response += f"**Risks:** {info['risks']}\n\n"
        response += "Would you like more information about what conditions this scan can detect?"
        return response

    def _check_symptom_patterns(self, message):
        """CHANGE 7: Fuzzy symptom pattern matching"""
        matched_categories = []

        for category, data in SYMPTOM_PATTERNS.items():
            if _fuzzy_match(message, data["keywords"]):
                matched_categories.append((category, data))

        if not matched_categories:
            return None

        if len(matched_categories) == 1:
            category, data = matched_categories[0]
            return self._format_symptom_response(category, data)

        # Multiple matches — combine
        response = "Based on your symptoms, multiple areas may be affected:\n\n"
        for category, data in matched_categories[:3]:
            response += f"**{category.title()}:** Possible conditions include {', '.join(data['conditions'][:3])}\n"
            response += f"💡 {data['advice']}\n\n"
        response += "⚠️ **Important:** This is not a diagnosis. Please consult with a healthcare professional for proper evaluation."
        return response

    def _format_symptom_response(self, category, data):
        """Format response for symptom-related queries"""
        response = f"Based on your {category} symptoms, possible conditions include:\n\n"
        for condition in data['conditions']:
            response += f"• {condition}\n"
        response += f"\n📋 **General advice:** {data['advice']}\n\n"
        response += "⚠️ **Important:** This is not a diagnosis. Please consult with a healthcare professional for proper evaluation.\n\n"
        response += "Would you like more information about any specific condition? Just ask \"What is [condition name]?\""
        return response

    def _check_specific_symptoms(self, message):
        """Check for specific symptom keywords"""
        for symptom, advice in HEALTH_KNOWLEDGE["general"]["symptoms"].items():
            symptom_words = symptom.replace("_", " ").split()
            if any(sw in message for sw in symptom_words) or symptom.replace("_", " ") in message:
                return (f"Regarding **{symptom.replace('_', ' ').title()}:**\n\n"
                        f"{advice}\n\n"
                        "⚠️ Please note: This is general information only. Always consult a healthcare professional for proper diagnosis and treatment.")
        return None

    def _get_prevention_advice(self, message):
        """Get prevention advice for conditions"""
        for condition, advice in HEALTH_KNOWLEDGE["general"]["prevention"].items():
            if condition in message or _fuzzy_match(message, [condition]):
                return f"**Prevention Tips for {condition.title()}:**\n\n{advice}\n\n⚠️ Consult your doctor for personalized prevention strategies."

        return ("**General Disease Prevention Tips:**\n\n"
                "• Maintain a healthy lifestyle with regular exercise\n"
                "• Eat a balanced diet rich in fruits and vegetables\n"
                "• Get regular health check-ups and screenings\n"
                "• Stay up to date on vaccinations\n"
                "• Don't smoke and limit alcohol consumption\n"
                "• Manage stress through mindfulness and relaxation\n"
                "• Get adequate sleep (7-9 hours for adults)\n"
                "• Stay hydrated — drink at least 8 glasses of water daily")

    def _check_lifestyle(self, message):
        """Check for lifestyle-related queries"""
        lifestyle_map = {
            "exercise": ["exercise", "workout", "fitness", "gym", "physical activity", "running", "walking"],
            "sleep": ["sleep", "insomnia", "rest", "sleeping", "tired at night"],
            "diet": ["diet", "eat", "food", "nutrition", "meal", "healthy eating", "weight loss"],
            "water": ["water", "hydration", "dehydrated", "thirsty", "fluid intake"]
        }

        for topic, keywords in lifestyle_map.items():
            if any(kw in message for kw in keywords):
                info = HEALTH_KNOWLEDGE["general"]["lifestyle"].get(topic, "")
                if info:
                    return f"**{topic.title()} Advice:**\n\n{info}\n\n💡 A healthy lifestyle is the foundation of disease prevention."

        return None

    def _fallback_response(self):
        """CHANGE 7: Improved fallback with helpful suggestions"""
        responses = [
            ("Sorry, I didn't quite understand that. Here are some things you can ask me:\n\n"
             "• \"What is glioma?\" — Disease explanations\n"
             "• \"I have a headache and nausea\" — Symptom analysis\n"
             "• \"Tell me about MRI scans\" — Imaging info\n"
             "• \"How to prevent heart disease?\" — Prevention tips\n"
             "• \"Exercise tips\" — Lifestyle advice\n\n"
             "Try rephrasing your question, or pick one of the topics above!"),
            ("I'm not sure I understand your question. I can help with:\n\n"
             "🔬 **Disease Info** — \"Explain pneumonia\"\n"
             "🩺 **Symptoms** — \"What causes chest pain?\"\n"
             "📷 **Scans** — \"What does a CT scan show?\"\n"
             "🛡️ **Prevention** — \"How to prevent diabetes?\"\n\n"
             "Please try rephrasing your question!"),
            ("I didn't catch that. Could you rephrase? Here are example questions:\n\n"
             "• \"What is a brain tumor?\"\n"
             "• \"I feel dizzy and nauseous\"\n"
             "• \"When should I get an X-ray?\"\n"
             "• \"Tips for healthy sleep\"\n\n"
             "I'm here to help with any health-related questions! 😊")
        ]
        return random.choice(responses)

    def get_conversation_history(self):
        """Get conversation history"""
        return self.conversation_history

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared. How can I help you today?"


# Create global instance
health_assistant = HealthAssistant()


def get_assistant_response(message):
    """Get response from health assistant"""
    return health_assistant.process_message(message)
