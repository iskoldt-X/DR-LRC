#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DR Audio Downloader

This script downloads a podcast episode from a given DR LYD URL.
It performs the following steps:
1.  Fetches the webpage for the DR LYD episode.
2.  Parses the page's JSON data to find the HLS (m3u8) audio stream URL.
3.  Uses the external tool 'ffmpeg' to download the stream and save it as a high-quality MP3 file.
4.  Creates a WAV version of the audio (16kHz, mono), which is the required format for many
    speech-to-text tools like Whisper.

Usage:
    python3 download_audio.py <DR_LYD_URL>

Example:
    python3 download_audio.py "https://www.dr.dk/lyd/p1/genstart/2024-05-16/da-israel-gik-i-graedekonernes-faelde"

"""

import json
import logging
import os
import re
import subprocess
import sys
import unicodedata

import requests
from bs4 import BeautifulSoup

# --- Configuration ---
# Configure logging to provide clear feedback to the user
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# --- Core Functions ---


def normalize_filename(filename: str) -> str:
    """
    Sanitizes a string to create a safe and valid filename.
    - Decomposes Unicode characters.
    - Removes characters that are not ASCII alphanumeric or dashes.
    - Replaces whitespace and other symbols with underscores.

    Args:
        filename: The original string to sanitize.

    Returns:
        A sanitized string suitable for use as a filename.
    """
    # Normalize to NFKD form to decompose characters like 'é' into 'e' and '´'
    nfkd_form = unicodedata.normalize("NFKD", filename)
    # Encode to ASCII, ignoring non-ASCII characters
    ascii_bytes = nfkd_form.encode("ASCII", "ignore")
    # Decode back to a string
    ascii_str = ascii_bytes.decode("ASCII")
    # Replace any sequence of non-alphanumeric characters (except dashes) with a single underscore
    sanitized = re.sub(r"[^A-Za-z0-9\-]+", "_", ascii_str)
    # Remove leading/trailing underscores that might result from the substitution
    sanitized = sanitized.strip("_")
    return sanitized


def convert_audio(input_file: str, output_file: str):
    """
    Uses ffmpeg to convert an audio file to a standard format for transcription.
    The output is a 16kHz, single-channel (mono) WAV file.

    Args:
        input_file: Path to the source audio file.
        output_file: Path where the converted WAV file will be saved.

    Raises:
        subprocess.CalledProcessError: If the ffmpeg command fails.
    """
    if not os.path.exists(input_file):
        logging.error(f"Input file for conversion not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logging.info(f"Converting '{input_file}' to '{output_file}'...")
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-i",
        input_file,
        "-ar",
        "16000",  # Set audio sampling rate to 16kHz
        "-ac",
        "1",  # Set audio channels to 1 (mono)
        "-c:a",
        "pcm_s16le",  # Set audio codec for WAV
        output_file,
    ]

    try:
        process = subprocess.run(
            command,
            check=True,
            capture_output=True,  # Capture stdout and stderr
            text=True,  # Decode output as text
        )
        logging.info(f"Successfully created WAV file: {output_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion failed for '{input_file}'.")
        logging.error(f"Command: {' '.join(command)}")
        # FFmpeg often prints crucial error details to stderr
        logging.error(f"FFmpeg Error Output:\n{e.stderr}")
        raise


def run_download(url: str) -> tuple[str, str] | None:
    """
    Runs the full download and conversion pipeline for a given DR LYD URL.

    Args:
        url: The DR LYD URL to process.

    Returns:
        A tuple of (mp3_filepath, wav_filepath) on success, otherwise None.
    """
    try:
        check_dependencies()
        logging.info(f"Starting audio download for URL: {url}")
        mp3_file, wav_file = download_dr_radio(url)
        return mp3_file, wav_file
    except Exception as e:
        # download_dr_radio already logs details. Just log the top-level failure.
        logging.error(f"Failed to complete download pipeline for URL '{url}'.")
        logging.debug(e, exc_info=True)
        return None


def download_dr_radio(url: str) -> tuple[str, str]:
    """
    Main function to handle the download and processing of a DR LYD URL.

    Args:
        url: The full URL of the DR LYD podcast episode.

    Returns:
        A tuple containing the filenames of the created (mp3_filename, wav_filename).

    Raises:
        RuntimeError: If the page structure is unexpected or audio links cannot be found.
        requests.exceptions.RequestException: If fetching the URL fails.
    """
    # 1. Fetch the HTML content of the page
    logging.info(f"Fetching page content from URL: {url}")
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch the URL '{url}'. Reason: {e}")
        raise

    # 2. Parse the HTML to find the hidden JSON data
    # DR websites often embed their initial page data in a <script> tag with id="__NEXT_DATA__"
    soup = BeautifulSoup(response.text, "html.parser")
    next_data_script = soup.find("script", id="__NEXT_DATA__")
    if not next_data_script:
        raise RuntimeError(
            "Could not find the __NEXT_DATA__ script block in the page. The site structure may have changed."
        )

    page_data = json.loads(next_data_script.string)

    # 3. Navigate the JSON structure to find the audio assets
    try:
        episode_details = page_data["props"]["pageProps"]["episode"]
        audio_assets = episode_details["audioAssets"]
        episode_title = episode_details.get("title", "dr_download")
    except KeyError:
        raise RuntimeError(
            "Failed to find 'episode' or 'audioAssets' in the page's JSON data. The site structure may have changed."
        )

    # 4. Find the HLS (HTTP Live Streaming) URL, which points to the audio stream manifest
    hls_asset_url = None
    for asset in audio_assets:
        if asset.get("format") == "HLS" and asset.get("url"):
            hls_asset_url = asset["url"]
            break

    if not hls_asset_url:
        raise RuntimeError("No HLS (m3u8) stream link found in the audio assets.")
    logging.info(f"Found HLS manifest API link: {hls_asset_url}")

    # 5. Get the final, redirect-resolved m3u8 URL
    try:
        with requests.get(
            hls_asset_url, allow_redirects=True, stream=True, timeout=15
        ) as r:
            r.raise_for_status()
            final_m3u8_url = r.url
    except requests.exceptions.RequestException as e:
        logging.error(
            f"Failed to resolve the final m3u8 URL from '{hls_asset_url}'. Reason: {e}"
        )
        raise
    logging.info(f"Resolved final m3u8 download URL: {final_m3u8_url}")

    # 6. Use ffmpeg to download the stream and save it as an MP3
    safe_title = normalize_filename(episode_title)
    mp3_filename = f"{safe_title}.mp3"

    logging.info(f"Starting download and conversion to MP3: '{mp3_filename}'...")
    ffmpeg_command = [
        "ffmpeg",
        "-y",  # Overwrite output without asking
        "-protocol_whitelist",
        "file,http,https,tcp,tls,crypto",  # Required for HLS streams
        "-i",
        final_m3u8_url,
        "-c:a",
        "libmp3lame",  # Use the LAME MP3 encoder
        "-q:a",
        "2",  # Set variable bitrate quality (0-9, lower is better)
        mp3_filename,
    ]

    try:
        process = subprocess.run(
            ffmpeg_command, check=True, capture_output=True, text=True
        )
        logging.info(f"Successfully created MP3 file: {mp3_filename}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg download failed for URL '{final_m3u8_url}'.")
        logging.error(f"Command: {' '.join(ffmpeg_command)}")
        logging.error(f"FFmpeg Error Output:\n{e.stderr}")
        raise

    # 7. Create the WAV version for transcription
    wav_filename = f"{safe_title}.wav"
    convert_audio(mp3_filename, wav_filename)

    return mp3_filename, wav_filename


def check_dependencies():
    """Checks for required external command-line tools."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True, text=True
        )
        logging.info("Dependency check passed: 'ffmpeg' is installed and accessible.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logging.error("------------------ DEPENDENCY ERROR ------------------")
        logging.error("FFmpeg was not found on your system.")
        logging.error("Please install FFmpeg and ensure it is in your system's PATH.")
        logging.error("Installation guide: https://ffmpeg.org/download.html")
        logging.error("----------------------------------------------------")
        sys.exit(1)


def main():
    """Main execution function."""
    # Check for correct command-line arguments
    if len(sys.argv) != 2 or not sys.argv[1].startswith("https://www.dr.dk/lyd/"):
        print(__doc__)  # Print the script's docstring for usage info
        sys.exit(1)

    url_to_download = sys.argv[1]

    results = run_download(url_to_download)

    if results:
        mp3_file, wav_file = results
        # --- Success Summary ---
        print("\n" + "=" * 50)
        logging.info("✅ Process completed successfully!")
        logging.info(f"  - MP3 file saved as: {mp3_file}")
        logging.info(f"  - WAV file for transcription saved as: {wav_file}")
        print("=" * 50)
    else:
        logging.error("\n" + "!" * 50)
        logging.error("❌ An unrecoverable error occurred during the process.")
        # The specific error details would have been logged already by the function that failed.
        logging.error("Please review the logs above for more details.")
        print("!" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
