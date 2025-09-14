#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LRC Lyrics Embedder for MP3 Files

This script embeds lyrics from one or more .lrc files into a corresponding MP3 file.
It creates a new copy of the MP3 for each language's LRC file to ensure the original
audio file remains untouched.

The script intelligently determines the language of the lyrics from the LRC filename
(e.g., "_english_translated.lrc") and embeds it with the correct ISO 639-2 language code.

Usage:
    python3 embed_lyrics.py <path_to_mp3_file> <path_to_lrc_file1> [<path_to_lrc_file2> ...]

Example (embedding a single LRC file):
    python3 embed_lyrics.py "my_podcast.mp3" "my_podcast.lrc"

Example (embedding original and translated LRC files):
    python3 embed_lyrics.py "my_podcast.mp3" "my_podcast.lrc" "my_podcast_chinese_translated.lrc"

Dependencies:
    - Python 3.6+
    - Python library: mutagen
      (Install with: pip install mutagen)
"""

import logging
import os
import re
import shutil
import sys

try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, USLT, Encoding
except ImportError:
    print("Error: The 'mutagen' library is not installed.")
    print("Please install it using: pip install mutagen")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Core Function (Directly from original script) ---


def run_embed_lyrics(mp3_file: str, lrc_file: str) -> str | None:
    """
    Embeds lyrics from a .lrc file into an MP3 file's ID3 USLT frame.
    This function creates a new MP3 file for the embedded lyrics.

    Args:
        mp3_file: The source MP3 file path.
        lrc_file: The source LRC file path.

    Returns:
        The path to the newly created MP3 file, or None on failure.
    """
    if not os.path.exists(mp3_file):
        logging.error(f"Source MP3 file not found: {mp3_file}")
        return None
    if not os.path.exists(lrc_file):
        logging.error(f"Source LRC file not found: {lrc_file}")
        return None

    # --- Determine Language and Output Filename ---
    lang_code = "dan"  # Default for original Danish lyrics
    lang_name_from_file = "original"  # Default description

    # Try to extract language from filenames like '_english_translated.lrc'
    match = re.search(r"_([a-z]{2,})_translated\.lrc$", lrc_file.lower())
    if match:
        lang_name_from_file = match.group(1)
        # Rudimentary mapping to 3-letter ISO 639-2 codes
        lang_map = {
            "english": "eng",
            "en": "eng",
            "chinese": "chi",
            "zh": "chi",
            "german": "ger",
            "de": "ger",
            "french": "fre",
            "fr": "fre",
            "spanish": "spa",
            "es": "spa",
            "danish": "dan",
            "da": "dan",
        }
        lang_code = lang_map.get(lang_name_from_file, "und")  # 'und' for undetermined

    # Create a unique output MP3 filename for this language version
    base_mp3_name = os.path.splitext(mp3_file)[0]
    output_mp3_file = f"{base_mp3_name}_{lang_name_from_file}.mp3"

    logging.info(f"Creating new MP3 with embedded lyrics: '{output_mp3_file}'")
    try:
        shutil.copyfile(mp3_file, output_mp3_file)
    except IOError as e:
        logging.error(f"Failed to create a copy of the MP3 file: {e}")
        return None

    # --- Embed Lyrics into the New MP3 Copy ---
    logging.info(
        f"Embedding '{lrc_file}' into '{output_mp3_file}' (language: {lang_code})..."
    )
    try:
        audio = MP3(output_mp3_file, ID3=ID3)
        if audio.tags is None:
            audio.add_tags()

        # Remove existing USLT frames to avoid duplicates
        # This is good practice even on a fresh copy
        audio.tags.delall("USLT")

        with open(lrc_file, "r", encoding="utf-8") as f:
            lyrics_text = f.read()

        # Add the new USLT frame
        audio.tags.add(
            USLT(
                encoding=Encoding.UTF8,
                lang=lang_code,
                desc="Lyrics",  # Description is standard, language code is key
                text=lyrics_text,
            )
        )
        audio.save(v2_version=3)  # Save as ID3v2.3 for wide compatibility
        logging.info(f"Successfully embedded lyrics into '{output_mp3_file}'")
        return output_mp3_file
    except Exception as e:
        logging.error(
            f"An error occurred while embedding lyrics into '{output_mp3_file}': {e}",
            exc_info=True,
        )
        if os.path.exists(output_mp3_file):
            os.remove(output_mp3_file)
        return None


def main():
    """Main execution function."""
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    mp3_input_path = sys.argv[1]
    lrc_file_paths = sys.argv[2:]  # Get all LRC files from arguments

    # --- Pre-run Checks ---
    if not mp3_input_path.lower().endswith(".mp3"):
        logging.error(
            f"Error: The first argument must be a path to an MP3 file. Got: '{mp3_input_path}'"
        )
        sys.exit(1)

    if not os.path.exists(mp3_input_path):
        logging.error(f"Input MP3 file not found: '{mp3_input_path}'")
        sys.exit(1)

    all_lrc_exist = True
    for lrc_path in lrc_file_paths:
        if not lrc_path.lower().endswith(".lrc"):
            logging.error(f"Error: Expected an LRC file, but got: '{lrc_path}'")
            all_lrc_exist = False
        elif not os.path.exists(lrc_path):
            logging.error(f"Input LRC file not found: '{lrc_path}'")
            all_lrc_exist = False

    if not all_lrc_exist:
        sys.exit(1)

    # --- Processing ---
    try:
        logging.info(
            f"Processing {len(lrc_file_paths)} LRC file(s) for '{mp3_input_path}'..."
        )
        output_files = []
        all_ok = True
        for lrc_path in lrc_file_paths:
            new_mp3 = run_embed_lyrics(mp3_input_path, lrc_path)
            if new_mp3:
                output_files.append(new_mp3)
            else:
                all_ok = False

        if all_ok:
            print("\n" + "=" * 50)
            logging.info("✅ Embedding process completed successfully!")
            for f in output_files:
                logging.info(f"  - Created file: {f}")
            print("=" * 50)
        else:
            logging.error("\n❌ An error occurred during the embedding process.")
            sys.exit(1)

    except Exception as e:
        logging.error(f"\n❌ An unrecoverable error occurred: {e}")
        logging.debug(e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
