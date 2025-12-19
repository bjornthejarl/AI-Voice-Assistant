"""
Sarah Voice Assistant - Web Interface with Memory & Chat History
Flask server with Whisper, Deepseek, Kokoro TTS, persistent memory and chat history
"""

import os
import io
import json
import tempfile
import base64
import numpy as np
import soundfile as sf
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from kokoro import KPipeline
import torch
from faster_whisper import WhisperModel

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    WHISPER_MODEL,
    TTS_VOICE,
    TTS_SAMPLE_RATE,
)
from tools import TOOL_DEFINITIONS, execute_tool

app = Flask(__name__)
CORS(app)

# Paths
MEMORY_FILE = os.path.join(os.path.dirname(__file__), 'sarah_memory.json')
HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'chat_history.json')

# Global instances
whisper_model = None
tts_pipeline = None
deepseek_client = None
conversation_history = []
memory = {}
chat_history = []  # All past conversations

# Smart system prompt with dynamic answer length
SMART_PROMPT = """You are Sarah, a thoughtful AI assistant.

## Response Length Guidelines:
- Simple questions (time, weather, yes/no): 1 sentence
- Explanations or how-to: 2-4 sentences  
- Complex topics or when user asks to elaborate: As much as needed
- Match the depth to the question's complexity

## Memory (facts about user):
{memory_context}

## Previous Conversations Summary:
{history_summary}

## Rules:
1. Always respond in English
2. Be natural and conversational
3. Reference remembered details when relevant
4. If user shares personal info, acknowledge and remember it
5. You can reference past conversations naturally when relevant"""


def load_memory():
    """Load memory from JSON file."""
    global memory
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, 'r') as f:
                memory = json.load(f)
            print(f"📚 Loaded memory: {len(memory)} entries")
    except Exception as e:
        print(f"⚠️ Could not load memory: {e}")
        memory = {}


def save_memory():
    """Save memory to JSON file."""
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump(memory, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save memory: {e}")


def load_chat_history():
    """Load chat history from JSON file."""
    global chat_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                chat_history = json.load(f)
            print(f"📜 Loaded {len(chat_history)} past conversations")
    except Exception as e:
        print(f"⚠️ Could not load history: {e}")
        chat_history = []


def save_chat_history():
    """Save chat history to JSON file."""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(chat_history, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save history: {e}")


def get_memory_context():
    """Format memory for the prompt."""
    if not memory:
        return "No stored facts about user yet."
    
    lines = []
    for key, value in memory.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def get_history_summary():
    """Get a summary of recent conversation topics."""
    if not chat_history:
        return "No previous conversations."
    
    # Get last 5 conversations
    recent = chat_history[-5:]
    summaries = []
    for conv in recent:
        date = conv.get('date', 'Unknown date')
        summary = conv.get('summary', 'General conversation')
        summaries.append(f"- {date}: {summary}")
    
    return "\n".join(summaries)


def summarize_conversation():
    """Summarize the current conversation before saving."""
    global conversation_history
    
    if len(conversation_history) <= 1:
        return None
    
    # Extract user messages for summary
    user_msgs = [m['content'] for m in conversation_history if m['role'] == 'user']
    if not user_msgs:
        return None
    
    # Create a simple summary (first topic + count)
    first_topic = user_msgs[0][:50] + "..." if len(user_msgs[0]) > 50 else user_msgs[0]
    summary = f"{first_topic} ({len(user_msgs)} exchanges)"
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'summary': summary,
        'messages': [
            {'role': m['role'], 'content': m.get('content', '')} 
            for m in conversation_history 
            if m['role'] in ['user', 'assistant'] and m.get('content')
        ]
    }


def update_memory_from_conversation(user_text, assistant_text):
    """Extract and store relevant information from conversation."""
    lower = user_text.lower()
    
    # Name detection
    if "my name is" in lower or "i'm called" in lower or "call me" in lower:
        for phrase in ["my name is", "i'm called", "call me"]:
            if phrase in lower:
                idx = lower.index(phrase) + len(phrase)
                name = user_text[idx:].strip().split()[0].strip('.,!?')
                if name and len(name) > 1:
                    memory["user_name"] = name.title()
                    save_memory()
                    break
    
    # Preferences
    if "i like" in lower or "i love" in lower:
        for phrase in ["i like", "i love"]:
            if phrase in lower:
                idx = lower.index(phrase)
                pref = user_text[idx:].split('.')[0].strip()
                if pref:
                    if "preferences" not in memory:
                        memory["preferences"] = []
                    if isinstance(memory.get("preferences"), list) and pref not in memory["preferences"]:
                        memory["preferences"].append(pref)
                        save_memory()
                break
    
    # Location
    if "i live in" in lower or "i'm from" in lower:
        for phrase in ["i live in", "i'm from"]:
            if phrase in lower:
                idx = lower.index(phrase) + len(phrase)
                location = user_text[idx:].strip().split()[0].strip('.,!?')
                if location and len(location) > 1:
                    memory["location"] = location.title()
                    save_memory()
                    break


def get_device():
    """
    Detect device: checks Docker's saved file first, then runtime detection.
    Docker saves device type at build time to .device_type file.
    """
    device_file = os.path.join(os.path.dirname(__file__), '.device_type')
    
    # Check if Docker saved the device type at build time
    if os.path.exists(device_file):
        with open(device_file, 'r') as f:
            saved_device = f.read().strip().lower()
            if saved_device == 'gpu':
                if torch.cuda.is_available():
                    return 'cuda'
                else:
                    print("⚠️ Docker built for GPU but CUDA unavailable, falling back to CPU")
                    return 'cpu'
            else:
                return 'cpu'
    
    # Runtime detection (outside Docker)
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def init_models():
    """Initialize AI models."""
    global whisper_model, tts_pipeline, deepseek_client
    
    print("🚀 Initializing Sarah Web Interface...")
    
    load_memory()
    load_chat_history()
    
    # Detect device (Docker saved file or runtime detection)
    device = get_device()
    compute_type = "float16" if device == "cuda" else "int8"
    
    # Whisper
    print("🎙️ Loading Whisper model...")
    whisper_model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
    if device == "cuda":
        print(f"   🎮 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   💻 Using CPU")
    
    # TTS
    print("🔊 Loading Kokoro TTS...")
    tts_pipeline = KPipeline(lang_code='a', device=device)
    
    # Deepseek
    print("🧠 Connecting to Deepseek...")
    deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    
    reset_conversation()
    print("✅ Ready!")


def reset_conversation():
    """Reset conversation with updated context."""
    global conversation_history
    prompt = SMART_PROMPT.format(
        memory_context=get_memory_context(),
        history_summary=get_history_summary()
    )
    conversation_history = [{"role": "system", "content": prompt}]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    global conversation_history, chat_history
    
    try:
        audio_data = request.json.get('audio')
        if not audio_data:
            return jsonify({'error': 'No audio'}), 400
        
        # Decode and transcribe
        audio_bytes = base64.b64decode(audio_data.split(',')[1] if ',' in audio_data else audio_data)
        
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        segments, _ = whisper_model.transcribe(temp_path, language="en")
        user_text = " ".join([seg.text for seg in segments]).strip()
        os.unlink(temp_path)
        
        if not user_text:
            return jsonify({'error': 'Could not transcribe'}), 400
        
        print(f"📝 User: {user_text}")
        
        # Refresh context if asking about name
        if memory.get("user_name") and "my name" in user_text.lower():
            reset_conversation()
        
        conversation_history.append({"role": "user", "content": user_text})
        
        # Get AI response
        response = deepseek_client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=conversation_history,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            max_tokens=500,
            temperature=0.7
        )
        
        message = response.choices[0].message
        
        # Handle tools
        if message.tool_calls:
            tool_results = []
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except:
                    args = {}
                print(f"🔧 Using: {tc.function.name}")
                result = execute_tool(tc.function.name, args)
                tool_results.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            
            conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
            })
            conversation_history.extend(tool_results)
            
            final = deepseek_client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=conversation_history,
                max_tokens=500,
                temperature=0.7
            )
            assistant_text = final.choices[0].message.content
        else:
            assistant_text = message.content
        
        # Clean any DSML/XML markup that the model might output
        if assistant_text:
            import re
            # Remove DSML-style markup
            assistant_text = re.sub(r'<[^>]*DSML[^>]*>.*?</[^>]*DSML[^>]*>', '', assistant_text, flags=re.IGNORECASE | re.DOTALL)
            assistant_text = re.sub(r'<[^>]*invoke[^>]*>.*?</[^>]*invoke[^>]*>', '', assistant_text, flags=re.IGNORECASE | re.DOTALL)
            assistant_text = re.sub(r'<[^>]*function_calls[^>]*>.*?</[^>]*function_calls[^>]*>', '', assistant_text, flags=re.IGNORECASE | re.DOTALL)
            assistant_text = assistant_text.strip()
            
            # If cleaning removed everything, provide fallback
            if not assistant_text:
                assistant_text = "I'll help you with that."
        
        conversation_history.append({"role": "assistant", "content": assistant_text})
        print(f"💬 Sarah: {assistant_text}")
        
        # Update memory
        update_memory_from_conversation(user_text, assistant_text)
        
        # Generate TTS
        generator = tts_pipeline(assistant_text, voice=TTS_VOICE)
        audio_segments = [audio for _, _, audio in generator]
        
        if audio_segments:
            combined = np.concatenate(audio_segments)
            buffer = io.BytesIO()
            sf.write(buffer, combined, TTS_SAMPLE_RATE, format='WAV')
            buffer.seek(0)
            audio_b64 = base64.b64encode(buffer.read()).decode()
            
            return jsonify({
                'text': assistant_text,
                'transcript': user_text,
                'audio': f'data:audio/wav;base64,{audio_b64}'
            })
        
        return jsonify({'text': assistant_text, 'transcript': user_text})
        
    except Exception as e:
        print(f"⚠️ Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/interrupt', methods=['POST'])
def interrupt():
    """Handle user interruption."""
    print("🛑 User interrupted")
    return jsonify({'status': 'ok'})


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset and save current conversation to history."""
    global chat_history
    
    # Save current conversation summary
    summary = summarize_conversation()
    if summary and len(summary.get('messages', [])) > 0:
        chat_history.append(summary)
        save_chat_history()
        print(f"💾 Saved conversation: {summary['summary']}")
    
    reset_conversation()
    return jsonify({'status': 'ok'})


@app.route('/api/memory', methods=['GET'])
def get_memory():
    return jsonify(memory)


@app.route('/api/history', methods=['GET'])
def get_history():
    return jsonify(chat_history)


@app.route('/api/memory', methods=['DELETE'])
def clear_memory():
    global memory
    memory = {}
    save_memory()
    reset_conversation()
    return jsonify({'status': 'cleared'})


@app.route('/api/history', methods=['DELETE'])
def clear_history():
    global chat_history
    chat_history = []
    save_chat_history()
    reset_conversation()
    return jsonify({'status': 'cleared'})


if __name__ == '__main__':
    init_models()
    print("\n🌐 Open http://localhost:5000 in your browser\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
