# Sarah - AI Voice Assistant

A voice assistant with a Jarvis-style web interface, powered by Whisper, Deepseek, and Kokoro TTS.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![GPU](https://img.shields.io/badge/GPU-CUDA%2012.1-green)

## Features

- 🎙️ **Local Speech Recognition** - Whisper (GPU accelerated)
- 🧠 **AI Responses** - Deepseek API with tool calling
- 🔊 **Natural TTS** - Kokoro text-to-speech
- 🌐 **Web Interface** - Animated Jarvis-style orb
- 💾 **Memory** - Remembers user info across sessions
- 🐳 **Docker Ready** - Auto GPU/CPU detection

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/bjornthejarl/AI-Voice-Assistant.git
cd AI-Voice-Assistant
pip install -r requirements.txt
```

2. Set up API key:
```bash
cp .env.example .env
# Edit .env with your Deepseek API key
```

3. Run:
```bash
python web_app.py
```

4. Open http://localhost:5000

## Docker

```bash
docker-compose up -d
```

## Usage

- **Click anywhere** or **press Space** to start talking
- Sarah auto-detects when you stop speaking
- Speak while she's talking to interrupt
- Remembers your name, preferences, and past conversations

## License

MIT
