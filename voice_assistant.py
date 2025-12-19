"""
Sarah - Voice Assistant with Whisper
Fast, local speech recognition with smart listening and interruptible playback.
"""

import os
import sys
import tempfile
import threading
import time
import json
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from kokoro import KPipeline
import pygame
import torch
from faster_whisper import WhisperModel
import webrtcvad

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    WAKE_WORD,
    WHISPER_MODEL,
    PREFERRED_MICROPHONE,
    SILENCE_DURATION,
    MAX_RECORDING_TIME,
    FOLLOW_UP_TIMEOUT,
    TTS_VOICE,
    TTS_SAMPLE_RATE,
    SYSTEM_PROMPT,
)

# Import tools
from tools import TOOL_DEFINITIONS, execute_tool


class VoiceAssistant:
    def __init__(self):
        print("🚀 Initializing Sarah Voice Assistant...")
        
        # Find preferred microphone
        self.mic_device = self._find_microphone()
        
        # Initialize Whisper for speech recognition
        print(f"🎙️ Loading Whisper '{WHISPER_MODEL}' model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        self.whisper = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
        if device == "cuda":
            print(f"🎮 Whisper using GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize VAD for voice activity detection (mode 1 = less aggressive)
        self.vad = webrtcvad.Vad(1)
        
        # Initialize Deepseek client
        print("🤖 Connecting to Deepseek API...")
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )
        
        # Initialize Kokoro TTS with GPU
        print("🔊 Loading Kokoro TTS engine...")
        self.tts_pipeline = KPipeline(lang_code='a', device=device)
        
        # Initialize pygame for audio playback
        pygame.mixer.init(frequency=TTS_SAMPLE_RATE)
        
        # State
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.is_speaking = False
        self.should_interrupt = False
        self.audio_queue = queue.Queue()
        
        print("✅ Sarah is ready!\n")

    def _find_microphone(self):
        """Find preferred microphone or use default."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if PREFERRED_MICROPHONE.lower() in dev['name'].lower() and dev['max_input_channels'] > 0:
                print(f"🎤 Found preferred microphone: {dev['name']}")
                return i
        print(f"⚠️ Preferred microphone '{PREFERRED_MICROPHONE}' not found, using default")
        return None

    def record_audio(self, timeout=None, listen_for_interrupt=False):
        """Record audio with smart silence detection using VAD."""
        sample_rate = 16000  # Whisper expects 16kHz
        frame_duration_ms = 30  # VAD frame size
        frame_size = int(sample_rate * frame_duration_ms / 1000)
        
        # Clear any stale audio from queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        audio_chunks = []
        silent_frames = 0
        max_silent_frames = int(SILENCE_DURATION * 1000 / frame_duration_ms)
        max_frames = int(MAX_RECORDING_TIME * 1000 / frame_duration_ms) if not timeout else int(timeout * 1000 / frame_duration_ms)
        speech_started = False
        
        # Timeout tracking
        start_time = time.time()
        
        def audio_callback(indata, frames, time_info, status):
            self.audio_queue.put(indata.copy())
        
        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='int16',
                device=self.mic_device,
                blocksize=frame_size,
                callback=audio_callback
            ):
                print("🎤 Listening... (speak now)")
                
                frame_count = 0
                last_speech_time = None  # Track actual time of last speech
                
                while frame_count < max_frames:
                    try:
                        chunk = self.audio_queue.get(timeout=0.1)
                        audio_chunks.append(chunk)
                        frame_count += 1
                        
                        # Convert to bytes for VAD
                        audio_bytes = chunk.tobytes()
                        
                        # Check for voice activity
                        audio_level = np.abs(chunk).mean()
                        
                        try:
                            vad_says_speech = self.vad.is_speech(audio_bytes, sample_rate)
                        except:
                            vad_says_speech = False
                        
                        # Consider it speech if audio level is above threshold OR VAD detects it
                        is_speech = audio_level > 80 or (vad_says_speech and audio_level > 30)
                        
                        if is_speech:
                            speech_started = True
                            last_speech_time = time.time()
                        elif speech_started and last_speech_time:
                            # Check if we've been silent for SILENCE_DURATION seconds
                            silence_time = time.time() - last_speech_time
                            if silence_time >= SILENCE_DURATION:
                                break
                        
                        # Check for interrupt during playback listening
                        if listen_for_interrupt and speech_started and self.is_speaking:
                            self.should_interrupt = True
                            break
                        
                        # Timeout check
                        if timeout and (time.time() - start_time) > timeout:
                            if not speech_started:
                                return None
                            break
                            
                    except queue.Empty:
                        continue
                        
        except Exception as e:
            print(f"⚠️ Recording error: {e}")
            return None
        
        if not speech_started:
            return None
        
        # Combine audio chunks
        audio_data = np.concatenate(audio_chunks).flatten().astype(np.float32) / 32768.0
        return audio_data

    def transcribe(self, audio_data):
        """Transcribe audio using Whisper."""
        if audio_data is None or len(audio_data) < 1600:  # Less than 0.1s
            return None
        
        print("🔄 Transcribing...")
        
        try:
            segments, info = self.whisper.transcribe(
                audio_data,
                beam_size=5,
                language="en",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            text = " ".join([seg.text for seg in segments]).strip()
            if text:
                print(f"📝 You said: '{text}'")
                return text
            return None
            
        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
            return None

    def listen_for_wake_word(self):
        """Listen for wake word using continuous recording."""
        print(f"💤 Listening for wake word: '{WAKE_WORD}'...")
        
        while True:
            try:
                audio = self.record_audio(timeout=3)
                if audio is not None:
                    text = self.transcribe(audio)
                    if text:
                        print(f"   Heard: '{text}'")
                        if WAKE_WORD.lower() in text.lower():
                            print("🎉 Wake word detected!")
                            return True
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return False

    def listen_for_input(self, timeout=None):
        """Listen for user input with smart silence detection."""
        audio = self.record_audio(timeout=timeout)
        if audio is None:
            print("⏱️ No speech detected")
            return None
        return self.transcribe(audio)

    def get_ai_response(self, user_text):
        """Get response from Deepseek API with function calling."""
        print("🧠 Thinking...")
        
        self.conversation_history.append({"role": "user", "content": user_text})
        
        try:
            response = self.client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=self.conversation_history,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=300,
                temperature=0.7
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                tool_results = []
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        arguments = json.loads(tool_call.function.arguments)
                    except:
                        arguments = {}
                    
                    print(f"🔧 Using: {tool_name}")
                    result = execute_tool(tool_name, arguments)
                    
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
                })
                self.conversation_history.extend(tool_results)
                
                # Get final response after tool use
                final_response = self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=self.conversation_history,
                    max_tokens=300,
                    temperature=0.7
                )
                assistant_message = final_response.choices[0].message.content
            else:
                assistant_message = message.content
            
            self.conversation_history.append({"role": "assistant", "content": assistant_message})
            print(f"💬 Sarah: {assistant_message}")
            return assistant_message
            
        except Exception as e:
            print(f"⚠️ API error: {e}")
            return "Sorry, I'm having trouble right now."

    def _speak_sentence(self, text):
        """Speak a single sentence quickly."""
        if not text or len(text) < 2:
            return
            
        self.is_speaking = True
        self.should_interrupt = False
        
        try:
            generator = self.tts_pipeline(text, voice=TTS_VOICE)
            audio_segments = []
            for _, _, audio in generator:
                audio_segments.append(audio)
            
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                # Use unique filename to avoid file locking
                import uuid
                temp_path = os.path.join(tempfile.gettempdir(), f"sarah_{uuid.uuid4().hex[:8]}.wav")
                sf.write(temp_path, combined_audio, TTS_SAMPLE_RATE)
                
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Start interrupt listener
                interrupt_thread = threading.Thread(target=self._listen_for_interrupt, daemon=True)
                interrupt_thread.start()
                
                while pygame.mixer.music.get_busy():
                    if self.should_interrupt:
                        pygame.mixer.music.stop()
                        print("\n🛑 Interrupted!")
                        break
                    time.sleep(0.03)
                
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            print(f"⚠️ TTS error: {e}")
        
        self.is_speaking = False

    def speak(self, text):
        """Speak full text (used for simple responses)."""
        print("🔊 Speaking...")
        self._speak_sentence(text)

    def _listen_for_interrupt(self):
        """Background listener for interruption - with noise filtering."""
        sample_rate = 16000
        frame_size = int(sample_rate * 0.03)
        
        # Use more aggressive VAD for interruption (3 = most aggressive)
        interrupt_vad = webrtcvad.Vad(3)
        
        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', device=self.mic_device, blocksize=frame_size) as stream:
                speech_frames = 0
                while self.is_speaking and not self.should_interrupt:
                    data, _ = stream.read(frame_size)
                    
                    # Check audio level first (noise gate)
                    audio_level = np.abs(data).mean()
                    if audio_level < 800:  # Below noise floor, skip
                        speech_frames = 0
                        continue
                    
                    # Then check VAD
                    try:
                        is_speech = interrupt_vad.is_speech(data.tobytes(), sample_rate)
                    except:
                        is_speech = audio_level > 1500  # High threshold fallback
                    
                    if is_speech:
                        speech_frames += 1
                        if speech_frames >= 10:  # ~300ms of sustained speech
                            self.should_interrupt = True
                            break
                    else:
                        speech_frames = max(0, speech_frames - 1)  # Gradual decay
        except:
            pass

    def play_activation_sound(self):
        """Quick activation beep."""
        duration = 0.1
        t = np.linspace(0, duration, int(TTS_SAMPLE_RATE * duration), False)
        beep = (np.sin(2 * np.pi * 880 * t) * 0.3).astype(np.float32)
        
        # Create temp file, write, close, then load
        temp_path = os.path.join(tempfile.gettempdir(), "sarah_beep.wav")
        sf.write(temp_path, beep, TTS_SAMPLE_RATE)
        pygame.mixer.music.load(temp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.02)
        try:
            os.unlink(temp_path)
        except:
            pass
        
        # Small delay to release audio device
        time.sleep(0.2)

    def run_conversation(self):
        """Handle conversation with interruption support."""
        self.play_activation_sound()
        self.conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        first_input = True
        while True:
            user_input = self.listen_for_input(timeout=None if first_input else FOLLOW_UP_TIMEOUT)
            first_input = False
            
            if user_input is None:
                if len(self.conversation_history) > 1:
                    print("👋 Returning to wake word mode.\n")
                    break
                else:
                    self.speak("How can I help?")
                    continue
            
            if any(p in user_input.lower() for p in ["goodbye", "bye", "that's all", "thanks"]):
                self.speak("Goodbye!")
                break
            
            response = self.get_ai_response(user_input)
            self.speak(response)
            print("🎤 Waiting for follow-up...")

    def run(self):
        """Main loop."""
        print("=" * 50)
        print("   🎙️  SARAH VOICE ASSISTANT (Whisper)  🎙️")
        print("=" * 50)
        print(f"Say '{WAKE_WORD}' to start. Press Ctrl+C to exit.\n")
        
        try:
            while True:
                if self.listen_for_wake_word():
                    self.run_conversation()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down Sarah. Goodbye!")
        finally:
            pygame.mixer.quit()


def main():
    if DEEPSEEK_API_KEY == "YOUR_API_KEY_HERE":
        print("⚠️ ERROR: Set your Deepseek API key in config.py")
        sys.exit(1)
    
    VoiceAssistant().run()


if __name__ == "__main__":
    main()
