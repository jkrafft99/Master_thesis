#!/usr/bin/env python3
import os
import sys
import glob
import whisper
import argparse
import torch

def transcribe_audio(audio_path, model_name="medium", language="nl", output_dir="whisper"):
    """Transcribe a single audio file using Whisper"""
    print(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)
    
    print(f"Transcribing: {os.path.basename(audio_path)}")
    # Run transcription with specified language
    result = model.transcribe(
        audio_path, 
        language=language,
        verbose=True
    )
    
    # Get the base filename without extension
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.txt")
    
    # Write the transcription to a file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    
    print(f"Transcription saved to: {output_path}")
    return output_path

def process_audio_files(input_dir, model_name, language, output_dir):
    """Process all audio files in a directory"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files (.wav, .mp3, .m4a, etc.)
    audio_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
    audio_files = []
    
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Check if CUDA is available for GPU acceleration
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    # Process each audio file
    for audio_path in audio_files:
        transcribe_audio(audio_path, model_name, language, output_dir)
    
    print(f"All transcriptions completed and saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper")
    parser.add_argument("--input", "-i", help="Input directory containing audio files", required=True)
    parser.add_argument("--output", "-o", help="Output directory for transcriptions", default="whisper")
    parser.add_argument("--model", "-m", help="Whisper model to use (tiny, base, small, medium, large)", default="medium")
    parser.add_argument("--language", "-l", help="Language code (nl for Dutch)", default="nl")
    
    args = parser.parse_args()
    
    process_audio_files(args.input, args.model, args.language, args.output)
