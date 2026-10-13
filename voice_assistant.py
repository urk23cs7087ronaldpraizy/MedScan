"""
Voice Assistant for MedScan AI with GPT Integration
Handles speech-to-text, GPT-4 analysis, and text-to-speech
"""

import openai
import speech_recognition as sr
import pyttsx3
import threading
import base64
import json
from io import BytesIO
from PIL import Image
import io

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

class VoiceAssistant:
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
        if openai_api_key:
            openai.api_key = openai_api_key
        
        # Configure TTS
        voices = tts_engine.getProperty('voices')
        tts_engine.setProperty('voice', voices[0].id if voices else None)
        tts_engine.setProperty('rate', 180)
    
    def transcribe_audio(self, audio_data):
        """
        Convert audio to text using speech recognition
        """
        try:
            # Use Google's speech recognition (free)
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"
    
    def get_gpt_response(self, prompt, context="healthcare"):
        """
        Get GPT response for health queries
        """
        if not self.openai_api_key:
            return "OpenAI API key not configured. Using fallback response."
        
        try:
            system_prompt = """
            You are MedScan AI Voice Assistant, a helpful medical health companion.
            Provide accurate, concise medical information based on the latest knowledge.
            Always disclaimer that you're not a doctor and recommend professional consultation.
            Use simple language, focus on symptoms, prevention, and when to seek help.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"GPT error: {e}"
    
    def speak_response(self, text):
        """
        Convert text to speech
        """
        def _speak():
            tts_engine.say(text)
            tts_engine.runAndWait()
        
        thread = threading.Thread(target=_speak)
        thread.daemon = True
        thread.start()
    
    def analyze_symptoms_voice(self, symptoms_text):
        """
        Analyze symptoms via voice with GPT
        """
        prompt = f"""
        Analyze these symptoms and provide:
        1. Most likely conditions (top 3)
        2. Urgency level (low/medium/high/emergency)
        3. Recommended actions
        4. When to seek immediate help
        
        Symptoms: {symptoms_text}
        """
        
        response = self.get_gpt_response(prompt)
        self.speak_response(response)
        return response
    
    def process_medical_image_voice(self, image_path=None, image_base64=None):
        """
        Process medical image via voice command
        """
        prompt = """
        I'm analyzing a medical image. Provide guidance on what to look for in X-ray/MRI/CT scans.
        Explain common findings and when to be concerned.
        """
        
        response = self.get_gpt_response(prompt)
        self.speak_response(response)
        return response

# Global instance (API key will be set later)
voice_assistant = VoiceAssistant()

def get_voice_response(audio_data, use_gpt=True):
    """
    Main function for voice processing
    """
    # Transcribe
    text = voice_assistant.transcribe_audio(audio_data)
    
    if use_gpt:
        # GPT analysis
        gpt_response = voice_assistant.get_gpt_response(text)
        voice_assistant.speak_response(gpt_response)
        return gpt_response
    else:
        # Fallback to rule-based
        from health_assistant import get_assistant_response
        response = get_assistant_response(text)
        voice_assistant.speak_response(response)
        return response

def set_openai_api_key(api_key):
    """Set OpenAI API key"""
    global voice_assistant
    voice_assistant = VoiceAssistant(openai_api_key=api_key)

