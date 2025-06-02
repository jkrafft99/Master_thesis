#!/usr/bin/env python3
import os
import sys
import glob
import argparse
from google.cloud import speech_v1p1beta1 as speech
import io
import wave

def transcribe_with_google(audio_path, language_code="nl-NL", output_dir="google"):
    """Transcribe audio using Google Cloud Speech-to-Text"""
    
    # Initialize the Google Speech client
    # Note: You need to set up authentication first
    client = speech.SpeechClient()

    # Load audio file
    with io.open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    # Configure audio settings
    audio = speech.RecognitionAudio(content=content)
    
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # Adjust based on your audio
        language_code=language_code,
        model="medical_conversation",  # Use medical model if available
        enable_automatic_punctuation=True,
        use_enhanced=True,  # Use enhanced model if available
    )

    # Perform the transcription
    print(f"Transcribing {os.path.basename(audio_path)} with Google Cloud Speech...")
    response = client.recognize(config=config, audio=audio)

    # Extract the transcription
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "

    # Save the transcription
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript.strip())
    
    print(f"Transcription saved to: {output_path}")
    return output_path

def process_audio_files(input_dir, output_dir, language_code):
    """Process all audio files in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files
    audio_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Process each audio file
    for audio_path in audio_files:
        try:
            transcribe_with_google(audio_path, language_code, output_dir)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files using Google Cloud Speech")
    parser.add_argument("--input", "-i", help="Input directory containing audio files", required=True)
    parser.add_argument("--output", "-o", help="Output directory for transcriptions", default="google")
    parser.add_argument("--language", "-l", help="Language code (nl-NL for Dutch)", default="nl-NL")
    
    args = parser.parse_args()
    
    # Check for Google Cloud credentials
    if not os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'):
        print("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        print("Please set up authentication following Google Cloud documentation")
        print("https://cloud.google.com/docs/authentication/getting-started")
    
    process_audio_files(args.input, args.output, args.language)
