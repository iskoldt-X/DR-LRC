#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Audio to Lyrics Generator (Final Version)

This script takes an audio file as input and generates original language (Danish) subtitles.
It uses the correct modern parameters for the newly compiled whisper-cli.

Usage:
    python3 generate_lyrics.py <path_to_audio_file> [--keep-srt] [--flash-attn]

Example:
    # Using CPU (most stable)
    python3 generate_lyrics.py "my_audio.wav"

    # Using GPU with Flash Attention (fastest)
    python3 generate_lyrics.py "my_audio.mp3" --flash-attn --keep-srt

Dependencies:
    - Python 3.6+
    - External command-line tools: ffmpeg, whisper-cli (must be compiled from source)
    - Environment Variable: WHISPER_MODEL_PATH must point to your GGML model file.
    - Environment Variable: WHISPER_CLI_PATH must point to your whisper-cli executable.
"""

import logging
import os
import re
import subprocess
import sys

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Core Functions ---

def convert_audio(input_file: str, output_file: str):
    """Uses ffmpeg to convert audio to the required format for Whisper (16kHz, mono WAV)."""
    logging.info(f"Converting '{input_file}' to WAV for transcription...")
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        output_file
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion failed for '{input_file}'.")
        logging.error(f"FFmpeg Error Output:\n{e.stderr}")
        raise

def generate_srt_with_whisper(wav_file: str, model_path: str, use_flash_attn: bool, language: str = "da", max_line_len: int = 120) -> str:
    """Uses the newly compiled whisper-cli to generate an SRT subtitle file."""
    base_name = os.path.splitext(wav_file)[0]
    srt_file = f"{base_name}.srt"
    
    whisper_executable = os.environ.get("WHISPER_CLI_PATH")
    if not whisper_executable or not os.path.exists(whisper_executable):
        logging.error("WHISPER_CLI_PATH environment variable is not set or the specified path is invalid.")
        raise FileNotFoundError("WHISPER_CLI_PATH environment variable must be set to the path of the whisper-cli executable.")

    cmd = [
        whisper_executable,
        "-m", model_path,
        "-f", wav_file,
        "-l", language,
        "-osrt",
        "-ml", str(max_line_len),
        "-of", base_name,
    ]

    if use_flash_attn:
        cmd.append("--flash-attn")
        logging.info("Running whisper-cli with Metal and Flash Attention enabled...")
    else:
        cmd.append("--no-gpu") # Use CPU for maximum stability
        logging.info("Running whisper-cli on CPU (most stable mode)...")

    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Give it ample time to process, especially for long audio files
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600) # 1 hour timeout
        logging.debug(f"Whisper-CLI progress log (stderr):\n{result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error("whisper-cli execution failed.")
        logging.error(f"Whisper-CLI Error Output (stderr):\n{e.stderr}")
        raise
    except subprocess.TimeoutExpired:
        logging.error("whisper-cli timed out after 1 hour.")
        raise

    if not os.path.exists(srt_file):
        raise FileNotFoundError(f"SRT file '{srt_file}' not found after whisper-cli execution.")

    logging.info(f"Successfully generated SRT file: {srt_file}")
    return srt_file

def srt_time_to_lrc(srt_time: str) -> str:
    """Converts SRT timestamp format to LRC format."""
    match = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", srt_time)
    if not match: return "[00:00.00]"
    h, m, s, ms = map(int, match.groups())
    total_minutes = h * 60 + m
    centiseconds = ms // 10
    return f"[{total_minutes:02}:{s:02}.{centiseconds:02}]"

def convert_srt_to_lrc(srt_file: str) -> str:
    """Converts a .srt file to .lrc format."""
    lrc_file = os.path.splitext(srt_file)[0] + ".lrc"
    logging.info(f"Converting '{srt_file}' to '{lrc_file}'...")
    with open(srt_file, "r", encoding="utf-8") as srt_f, open(lrc_file, "w", encoding="utf-8") as lrc_f:
        content = srt_f.read().strip()
        blocks = re.split(r'\n\s*\n', content)
        for block in blocks:
            lines = block.strip().splitlines()
            if len(lines) >= 2 and "-->" in lines[1]:
                start_time_str = lines[1].split(" --> ")[0]
                text = " ".join(lines[2:])
                lrc_f.write(f"{srt_time_to_lrc(start_time_str)}{text}\n")
    logging.info(f"Successfully generated LRC file: {lrc_file}")
    return lrc_file

def run_generate_lyrics(
    input_audio_path: str, use_flash_attn: bool, cleanup_srt: bool = True
) -> str | None:
    """
    Generates an LRC file from an audio file, with options for cleanup.

    Args:
        input_audio_path: Path to the input audio file (e.g., .mp3, .wav).
        use_flash_attn: Whether to use Flash Attention with whisper-cli.
        cleanup_srt: If True, the intermediate SRT file will be deleted.

    Returns:
        The path to the generated .lrc file on success, otherwise None.
    """
    model_path = os.environ.get("WHISPER_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        logging.error("WHISPER_MODEL_PATH environment variable is not set or invalid.")
        return None

    if not os.path.exists(input_audio_path):
        logging.error(f"Input file not found: '{input_audio_path}'")
        return None

    wav_for_processing = None
    created_temp_wav = False
    srt_file_path = None

    try:
        if not input_audio_path.lower().endswith(".wav"):
            base_name = os.path.splitext(input_audio_path)[0]
            wav_for_processing = f"{base_name}.wav"
            convert_audio(input_audio_path, wav_for_processing)
            created_temp_wav = True
        else:
            wav_for_processing = input_audio_path

        srt_file_path = generate_srt_with_whisper(
            wav_for_processing, model_path, use_flash_attn
        )
        lrc_file_path = convert_srt_to_lrc(srt_file_path)

        return lrc_file_path

    except Exception as e:
        logging.error(f"\n❌ An unrecoverable error occurred during the lyric generation process.")
        logging.debug(e, exc_info=True)
        return None
    finally:
        # Cleanup intermediate files
        files_to_remove = []
        if srt_file_path and os.path.exists(srt_file_path) and cleanup_srt:
            files_to_remove.append(srt_file_path)
        if created_temp_wav and os.path.exists(wav_for_processing):
            files_to_remove.append(wav_for_processing)

        if files_to_remove:
            logging.info("Cleaning up intermediate files...")
            for f in files_to_remove:
                try:
                    os.remove(f)
                    logging.info(f"  - Removed: {f}")
                except OSError as e:
                    logging.warning(f"Could not remove '{f}': {e}")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_audio_path = sys.argv[1]
    keep_srt = "--keep-srt" in sys.argv
    use_flash_attn = "--flash-attn" in sys.argv

    lrc_file_path = run_generate_lyrics(
        input_audio_path=input_audio_path,
        use_flash_attn=use_flash_attn,
        cleanup_srt=not keep_srt,
    )

    if lrc_file_path:
        print("\n" + "="*50)
        logging.info("✅ Transcription process completed successfully!")
        logging.info(f"  - Final LRC file created: {lrc_file_path}")
        if keep_srt:
            # Reconstruct srt path for logging, as the run_... function cleans it up
            srt_path = os.path.splitext(lrc_file_path)[0] + ".srt"
            logging.info(f"  - Intermediate SRT file kept: {srt_path}")
        print("="*50)
    else:
        # Error is already logged by the pipeline function
        sys.exit(1)


if __name__ == "__main__":
    main()