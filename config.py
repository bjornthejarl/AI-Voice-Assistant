# Voice Assistant Configuration
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Deepseek API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "YOUR_API_KEY_HERE")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Wake Word Settings
WAKE_WORD = "hey siri"

# Whisper Speech Recognition Settings
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium
PREFERRED_MICROPHONE = "GK65"  # Will use this if found, otherwise default
SILENCE_DURATION = 3.5  # Seconds of silence before stopping (longer = waits for you)
MAX_RECORDING_TIME = 30  # Maximum seconds for a single phrase
FOLLOW_UP_TIMEOUT = 5  # Seconds to wait for follow-up

# TTS Settings
TTS_VOICE = "af_heart"  # Female voice for Sarah
TTS_SAMPLE_RATE = 24000

# System Prompt - Optimized for short responses
SYSTEM_PROMPT = """You are Sarah, a voice assistant. Rules:
1. ALWAYS respond in English
2. Be VERY concise - max 1-2 sentences
3. No filler words, get to the point
4. Only elaborate if explicitly asked"""

# Optional: Weather API Key
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY", "")
