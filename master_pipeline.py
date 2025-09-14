#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Master Audio Processing Pipeline

This script orchestrates the entire audio processing workflow by importing and
calling functions from the refactored modular scripts. It can handle both
DR LYD URLs and local audio files.

Workflow:
1.  Get Audio: Downloads from a URL or prepares a local file.
2.  Generate Lyrics: Transcribes the audio to an LRC file.
3.  Embed Lyrics: Embeds the original language LRC into a copy of the MP3.
4.  Translate (Optional): Translates the LRC file to a target language.
5.  Embed Translation (Optional): Embeds the translated LRC into another MP3 copy.

Usage:
    python3 master_pipeline.py <DR_LYD_URL or audio_file_path> [target_language]

Example (Download & Transcribe):
    python3 master_pipeline.py "https://www.dr.dk/lyd/p1/genstart/..."

Example (Download, Transcribe & Translate):
    python3 master_pipeline.py "https://www.dr.dk/lyd/p1/genstart/..." English

Example (Local file, Transcribe & Translate):
    python3 master_pipeline.py "my_podcast.mp3" Chinese
"""

import logging
import os
import sys

# Import the refactored pipeline functions
from download_audio import run_download
from generate_lyrics import run_generate_lyrics
from translate_lrc import run_translate_lrc
from embed_lyrics import run_embed_lyrics
from dr_lrc_gemini import convert_audio # Re-use a utility if needed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_arg = sys.argv[1]
    target_language = sys.argv[2] if len(sys.argv) > 2 else None

    mp3_for_embedding = None
    audio_for_transcription = None

    # --- Step 1: Get Audio File ---
    logging.info("--- Step 1: Preparing Audio ---")
    if "dr.dk" in input_arg:
        download_result = run_download(input_arg)
        if not download_result:
            logging.error("Audio download failed. Aborting pipeline.")
            sys.exit(1)
        mp3_for_embedding, audio_for_transcription = download_result
    else:
        if not os.path.exists(input_arg):
            logging.error(f"Local file not found: {input_arg}")
            sys.exit(1)
        
        input_ext = os.path.splitext(input_arg)[1].lower()
        base_name = os.path.splitext(input_arg)[0]

        if input_ext == ".mp3":
            mp3_for_embedding = input_arg
            audio_for_transcription = input_arg # generate_lyrics handles conversion
        elif input_ext == ".wav":
            audio_for_transcription = input_arg
            mp3_for_embedding = f"{base_name}.mp3"
            logging.info(f"Input is WAV, creating MP3 for embedding: {mp3_for_embedding}")
            convert_audio(audio_for_transcription, mp3_for_embedding)
        else:
            logging.error(f"Unsupported local file type: {input_arg}. Please provide an MP3 or WAV file.")
            sys.exit(1)

    logging.info(f"Audio ready: MP3 for embedding at '{mp3_for_embedding}', Audio for transcription at '{audio_for_transcription}'")

    # --- Step 2: Generate Original Lyrics ---
    logging.info("\n--- Step 2: Generating Original Lyrics (LRC) ---")
    original_lrc_path = run_generate_lyrics(audio_for_transcription, use_flash_attn=True, cleanup_srt=True)
    if not original_lrc_path:
        logging.error("Lyric generation failed. Aborting pipeline.")
        sys.exit(1)
    logging.info(f"Original LRC file created: {original_lrc_path}")

    # --- Step 3: Embed Original Lyrics ---
    logging.info("\n--- Step 3: Embedding Original Lyrics ---")
    original_mp3_output = run_embed_lyrics(mp3_for_embedding, original_lrc_path)
    if not original_mp3_output:
        logging.warning("Failed to embed original lyrics.")
    else:
        logging.info(f"Created MP3 with original lyrics: {original_mp3_output}")

    # --- Step 4 & 5: Translate and Embed if requested ---
    if target_language:
        logging.info(f"\n--- Step 4: Translating Lyrics to {target_language} ---")
        translated_lrc_path = run_translate_lrc(original_lrc_path, target_language)
        if not translated_lrc_path:
            logging.error("Translation failed. Aborting further steps.")
            sys.exit(1)
        logging.info(f"Translated LRC file created: {translated_lrc_path}")

        logging.info("\n--- Step 5: Embedding Translated Lyrics ---")
        translated_mp3_output = run_embed_lyrics(mp3_for_embedding, translated_lrc_path)
        if not translated_mp3_output:
            logging.warning("Failed to embed translated lyrics.")
        else:
            logging.info(f"Created MP3 with translated lyrics: {translated_mp3_output}")

    logging.info("\n--- ✅ Master Pipeline Finished ---")

if __name__ == "__main__":
    main()