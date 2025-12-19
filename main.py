import soundfile as sf
from kokoro import KPipeline

# Initialize the TTS pipeline (American English)
pipeline = KPipeline(lang_code='a')

# Text to convert to speech
text = '''
Hello, how are you? My name is Sarah, how may I help you today?'''

# Generate audio using the 'af_heart' voice
generator = pipeline(text, voice='af_heart')

# Process and save each audio segment
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Segment {i}: {gs}")
    output_file = f'output_{i}.wav'
    sf.write(output_file, audio, 24000)
    print(f"Saved: {output_file}")

print("\nDone! Audio files saved.")