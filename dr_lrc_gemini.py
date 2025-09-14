import unicodedata
import re
import json
import subprocess
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import os
import sys
import shutil  # For copying mp3 to a new filename
import time
import random
import logging

# --- Imports for Gemini API translation ---
from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, TooManyRequests


from mutagen.mp3 import MP3
from mutagen.id3 import ID3, USLT, Encoding

# ────────────── Gemini API Config ──────────────
# Provide a comma‑separated list of keys in the GEMINI_API_KEYS env‑var
GEMINI_API_KEYS_ENV_VAR = "GEMINI_API_KEYS"
GEMINI_API_KEYS = [
    k.strip() for k in os.getenv(GEMINI_API_KEYS_ENV_VAR, "").split(",") if k.strip()
]

# Only configure client if keys are present, to allow non-translation use
_client_gemini = None
if GEMINI_API_KEYS:
    _current_key_index_gemini = 0
    _request_count_gemini = 0
    _client_gemini = genai.Client(api_key=GEMINI_API_KEYS[_current_key_index_gemini])
else:
    # This allows the script to run if no keys are set AND no translation is requested.
    # A check will be performed later if translation is actually attempted.
    pass

MAX_PER_API_GEMINI = int(
    os.getenv("MAX_PER_API_GEMINI", "50")
)  # Higher limit for Gemini Flash
MAX_RETRIES_GEMINI = 20
BASE_DELAY_GEMINI = 15  # Seconds

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


def _get_client_gemini() -> genai.Client:
    """Return a Gemini client bound to the current key; rotate when quota exhausted."""
    global _current_key_index_gemini, _request_count_gemini, _client_gemini

    if not GEMINI_API_KEYS:  # Should be caught before calling this, but as a safeguard
        raise RuntimeError(
            f"Gemini API keys not configured. Set {GEMINI_API_KEYS_ENV_VAR}."
        )

    if _request_count_gemini >= MAX_PER_API_GEMINI:
        _current_key_index_gemini = (_current_key_index_gemini + 1) % len(
            GEMINI_API_KEYS
        )
        _client_gemini = genai.Client(
            api_key=GEMINI_API_KEYS[_current_key_index_gemini]
        )
        _request_count_gemini = 0
        logging.info(
            f"🔄 Switched to Gemini API key #{_current_key_index_gemini + 1}/{len(GEMINI_API_KEYS)}."
        )
    return _client_gemini


def _count_request_gemini():
    """Increment the usage counter after each *successful* completion for Gemini."""
    global _request_count_gemini
    _request_count_gemini += 1


def normalize_filename(filename):
    # Normalize to NFKD form to decompose characters
    nfkd_form = unicodedata.normalize("NFKD", filename)
    # Encode to ASCII bytes, ignoring characters that can't be converted
    ascii_bytes = nfkd_form.encode("ASCII", "ignore")
    # Decode back to a string
    ascii_str = ascii_bytes.decode("ASCII")
    # Replace any non-alphanumeric characters (except dash) with underscores
    ascii_str = re.sub(r"[^A-Za-z0-9\-]+", "_", ascii_str)
    return ascii_str


#################################
# Translation-related utilities
#################################
def try_parse_json(raw_str):
    """
    Try to parse the string into JSON. If it fails, return None.
    """
    if not raw_str:  # Handle case where raw_str might be None
        return None
    raw_str = raw_str.strip()
    # Sometimes the model outputs ```json ... ``` so perform simple stripping
    raw_str = re.sub(r"^```(json)?|```$", "", raw_str, flags=re.IGNORECASE).strip()

    try:
        data = json.loads(raw_str)
        return data
    except json.JSONDecodeError:
        return None


def _build_gemini_response_schema_for_dict(keys_list: list[str]) -> dict:
    """
    Builds a Gemini response schema for a JSON object where all values are strings
    and all specified keys are required.
    Example: keys_list = ["0", "1", "text"] ->
    {
        "type": "object",
        "properties": {
            "0": {"type": "string"},
            "1": {"type": "string"},
            "text": {"type": "string"}
        },
        "required": ["0", "1", "text"]
    }
    """
    properties = {key: {"type": "string"} for key in keys_list}
    return {
        "type": "object",
        "properties": properties,
        "required": keys_list,
    }


def batch_translate_dict_mode_gemini(
    danish_texts, target_language="English", debug=False
):
    """
    Wrap multiple lines of Danish into a dict, have the Gemini model output JSON,
    and parse it. Uses Gemini API with API key rotation and specific error handling.
    STRICTLY ADHERES to Anki script's way of using GenerateContentConfig.
    """
    if not _client_gemini:  # Check if client was initialized (i.e. keys were provided)
        logging.error(
            f"Gemini API client not initialized. Please set {GEMINI_API_KEYS_ENV_VAR}."
        )
        return {}

    # 1) Prepare the data structure we want to send to the model
    data_to_translate = {str(i): text for i, text in enumerate(danish_texts)}
    expected_keys = list(data_to_translate.keys())

    # 2) Construct the system prompt (remains largely the same)
    system_prompt_content = f"""
        You are an expert translation model specifically optimized for translating Danish podcast subtitles into {target_language}.
        Your ONLY task is to translate the given JSON dictionary's values from Danish into accurate, fluent, and contextually appropriate {target_language} suitable for audio subtitles.

        Requirements:
        - Return ONLY a valid JSON object with exactly the SAME keys and translated values, also same number of keys.
        - The keys in the output JSON MUST be: {', '.join(f'"{k}"' for k in expected_keys)}.
        - Do NOT add any extra fields or metadata.
        - Ensure translations are natural, conversational, and appropriate for spoken audio subtitles.
        - Preserve proper nouns, titles, place names, and culturally specific references without translation unless widely known equivalents exist.
        - Absolutely avoid literal unicode escapes (such as \\uXXXX). Output must be human-readable characters.
        - Avoid repeating the same translation for multiple distinct inputs unless identical in Danish.
        - Maintain consistency in tone and style across all subtitles.
        """

    # 3) Construct the user prompt (remains largely the same)
    user_prompt_content = f"""
        Translate the following JSON dictionary containing Danish podcast subtitles into {target_language}, exactly preserving the keys and returning valid JSON as specified.

        Input JSON:
        {json.dumps(data_to_translate, ensure_ascii=False, indent=2)}

        Output requirements:
        - Output JSON must exactly match the input JSON structure (same keys: {', '.join(f'"{k}"' for k in expected_keys)}, translated values, same number of keys).
        - Each translation should be suitable for subtitle reading, natural sounding, concise, and contextually accurate for spoken content.
        - DO NOT include explanations, commentary, or literal unicode escapes (\\uXXXX sequences).
        - Ensure cultural nuances or references remain clear to an audience unfamiliar with Danish culture by briefly adapting or clarifying if necessary.

        Provide the translated JSON ONLY:
        """

    # 3b) Define the response schema for Gemini
    gemini_schema = _build_gemini_response_schema_for_dict(expected_keys)

    # 4) Send up to max_retries requests to the model
    last_feedback = None
    for attempt in range(MAX_RETRIES_GEMINI):
        if attempt == MAX_RETRIES_GEMINI - 1:
            logging.warning(
                f"[WARNING] Maximum retries ({MAX_RETRIES_GEMINI}) reached. You may see incorrect translations for batch: {danish_texts[:2]}..."
            )

        current_user_prompt = user_prompt_content
        if last_feedback:
            current_user_prompt = f"{user_prompt_content}\n\nPREVIOUS ATTEMPT FAILED: {last_feedback}. PLEASE CORRECT AND ADHERE STRICTLY TO THE SCHEMA AND REQUIREMENTS."

        client = _get_client_gemini()
        raw_output = None
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=[current_user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt_content,
                    temperature=1,
                    max_output_tokens=60000,  # Anki script used 8000
                    response_mime_type="application/json",
                    response_schema=gemini_schema,
                ),
            )
            raw_output = response.text
            _count_request_gemini()  # Count successful API call

        except (ResourceExhausted, TooManyRequests) as e:
            logging.warning(
                f"Gemini API rate limit or quota error (attempt {attempt + 1}/{MAX_RETRIES_GEMINI}): {e}. Switching key and retrying after delay..."
            )
            _request_count_gemini = (
                MAX_PER_API_GEMINI  # Force key switch on next _get_client_gemini()
            )
            time.sleep(
                BASE_DELAY_GEMINI * (attempt + 1)
                + random.uniform(0, BASE_DELAY_GEMINI / 2)
            )  # Add jitter
            last_feedback = f"API limit error: {e}"
            if attempt == MAX_RETRIES_GEMINI - 1:
                logging.error(
                    f"Failed on API limit after max retries for batch: {danish_texts[:2]}..."
                )
            continue
        except Exception as e:
            logging.error(
                f"Gemini API request failed (attempt {attempt + 1}/{MAX_RETRIES_GEMINI}): {type(e).__name__} - {e}"
            )
            time.sleep(
                BASE_DELAY_GEMINI * (attempt + 1)
                + random.uniform(0, BASE_DELAY_GEMINI / 2)
            )  # Add jitter
            last_feedback = f"General API error: {type(e).__name__} - {e}"
            if attempt == MAX_RETRIES_GEMINI - 1:
                logging.error(
                    f"Failed on General API error after max retries for batch: {danish_texts[:2]}..."
                )
            continue

        if debug:
            logging.info(
                f"\n[DEBUG] Attempt {attempt+1} - Model raw output:\n{raw_output}\n"
            )

        # 4a) Check if the raw text contains literal "\uXXXX"
        if raw_output and re.search(r"\\u[0-9A-Fa-f]{4}", raw_output):
            if debug:
                logging.info(
                    "[DEBUG] Detected literal \\u escapes in raw output, retrying...\n"
                )
            last_feedback = "Output contained literal \\uXXXX escapes."
            continue

        # 4b) Attempt to parse the JSON
        parsed = try_parse_json(raw_output)
        if parsed is not None and isinstance(parsed, dict):
            if set(parsed.keys()) != set(data_to_translate.keys()):
                if debug:
                    logging.info(
                        f"[DEBUG] Keys mismatch. Expected: {set(data_to_translate.keys())}, Got: {set(parsed.keys())}. Retrying...\n"
                    )
                last_feedback = f"The number of keys or key names in the translated JSON ({set(parsed.keys())}) does not match the input JSON ({set(data_to_translate.keys())}). Expected keys: {expected_keys}"
                continue

            # Additional check: verify that the same translation is not repeated for more than 2 distinct inputs.
            translation_groups = defaultdict(list)
            for key, translation in parsed.items():
                translation_groups[translation].append(key)

            suspicious = False
            for translation, keys_with_same_translation in translation_groups.items():
                if (
                    len(keys_with_same_translation) > 2
                ):  # If same translation for more than 2 items
                    # Check if the corresponding original Danish texts for these keys are not all identical.
                    original_texts_for_these_keys = {
                        data_to_translate[k]
                        for k in keys_with_same_translation
                        if k in data_to_translate
                    }
                    if (
                        len(original_texts_for_these_keys) > 1
                    ):  # More than one unique Danish source text got the same translation
                        suspicious = True
                        if debug:
                            logging.info(
                                f"[DEBUG] Detected suspicious repeated translation for keys {keys_with_same_translation} ('{translation}') from distinct originals: {original_texts_for_these_keys}"
                            )
                        break
            if suspicious:
                if debug:
                    logging.info(
                        "[DEBUG] Detected suspicious repeated translations for different input texts in batch, retrying...\n"
                    )
                last_feedback = "Suspicious repeated translation for different input texts detected. Please provide distinct translations for distinct inputs unless the inputs are truly synonymous in context."
                continue

            # If JSON parse succeeded and all checks passed, return the result.
            return parsed
        else:
            if debug:
                logging.info(
                    f"[DEBUG] JSON parse failed or not a dict (raw: '{raw_output}') - retrying...\n"
                )
            last_feedback = f"Output was not valid JSON or not a dictionary. Raw output (first 200 chars): '{str(raw_output)[:200]}...'"

    # 5) If we used all retries without success, return an empty dict
    logging.error(
        f"Failed to translate batch after {MAX_RETRIES_GEMINI} attempts. Last known issue: {last_feedback}. Batch started with: {danish_texts[:2]}..."
    )
    return {}


def process_lrc_file_batch_dict(
    input_file, output_file, target_language="English", batch_size=30, debug=False
):
    """
    Read a .lrc file, translate its content in batches using the Gemini "JSON dict" method,
    and insert the translation into a new LRC file as: original / translation.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract all lines that need translation.
    translation_indices = []
    danish_texts = []
    for idx, line in enumerate(lines):
        match = re.match(r"(\[\d{2}:\d{2}\.\d{2}\])(.*)", line.strip())
        if match:
            timestamp, danish_line = match.groups()
            if danish_line.strip():  # Only add lines with actual text
                translation_indices.append(idx)
                danish_texts.append(danish_line.strip())

    translations = []
    # Translate in batches.
    for i in range(0, len(danish_texts), batch_size):
        batch = danish_texts[i : i + batch_size]
        if (
            not batch
        ):  # Skip if batch is empty (shouldn't happen with current logic but good check)
            continue

        if debug:
            logging.info(
                f"\n[DEBUG] Processing batch {i // batch_size + 1} of { (len(danish_texts) + batch_size -1) // batch_size } ({len(batch)} items): First item '{batch[0]}'"
            )

        # Using the new Gemini-based function
        result_dict = batch_translate_dict_mode_gemini(
            batch, target_language=target_language, debug=debug
        )

        # Extract translations in order.
        # Ensure result_dict has translations for all items in the batch, even if empty
        for j in range(len(batch)):
            key_str = str(j)
            translation_value = result_dict.get(
                key_str,
                f"Error: Translation for item {j} missing",  # Make missing translation obvious
            )
            translations.append((translation_value or "").strip())

    # Ensure we have a translation (even if empty or error message) for every Danish text
    if len(translations) != len(danish_texts):
        logging.warning(
            f"Translation count mismatch: expected {len(danish_texts)}, got {len(translations)}. Some lines might not be translated or have errors."
        )
        # Pad with error messages if necessary
        translations.extend(
            [f"Error: Missing translation pad"]
            * (len(danish_texts) - len(translations))
        )

    # Insert the translation results into the file lines.
    translated_lines = lines.copy()  # Work on a copy
    current_translation_idx = 0
    for original_line_idx in translation_indices:
        match = re.match(
            r"(\[\d{2}:\d{2}\.\d{2}\])(.*)", lines[original_line_idx].strip()
        )
        if match:
            timestamp, original_text_content = match.groups()
            if current_translation_idx < len(translations):
                current_translation = translations[current_translation_idx]
                translated_lines[original_line_idx] = (
                    f"{timestamp}{original_text_content.strip()} / {current_translation}\n"
                )
                current_translation_idx += 1
            else:
                # This case should be prevented by the padding above, but as a safeguard:
                logging.error(
                    f"Missing translation for line index {original_line_idx} during reconstruction."
                )
                translated_lines[original_line_idx] = (
                    f"{timestamp}{original_text_content.strip()} / Error: No translation available\n"
                )
        else:
            # This line was not supposed to be translated, keep as is (already in translated_lines copy)
            pass

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(translated_lines)

    logging.info(f"Translation completed. Output saved to: {output_file}")


def convert_audio(input_file, output_file):
    ext = output_file.split(".")[-1].lower()
    cmd = ["ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-ac", "1"]

    if ext == "wav":
        cmd += ["-c:a", "pcm_s16le"]
    elif ext == "mp3":
        cmd += ["-c:a", "libmp3lame", "-q:a", "2"]  # Good quality VBR
    elif ext == "flac":
        cmd += ["-c:a", "flac"]
    else:
        # If not a known audio extension for specific codec, let ffmpeg decide
        logging.warning(
            f"Unknown output extension {ext} for convert_audio. Letting ffmpeg decide codec."
        )
        pass

    cmd.append(output_file)

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # Capture output
        )
        logging.info(f"Conversion complete: {output_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion failed for {input_file} to {output_file}:")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Stdout: {e.stdout.decode(errors='ignore')}")
        logging.error(f"Stderr: {e.stderr.decode(errors='ignore')}")
        raise  # Re-raise the exception to halt execution if critical


#########################################################
# 1) Download from DR LYD -> MP3, then convert MP3 -> WAV
#########################################################
def download_dr_radio(url):
    """
    Given a full DR LYD URL, extract the m3u8 link from the page, use ffmpeg to download/convert to MP3,
    then create a WAV version for Whisper processing. Returns (mp3_filename, wav_filename).
    """
    # Request the page
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch DR URL {url}: {e}")
        raise
    html = resp.text

    # Use BeautifulSoup to find the <script id="__NEXT_DATA__"> block
    soup = BeautifulSoup(html, "html.parser")
    next_data_script = soup.find("script", attrs={"id": "__NEXT_DATA__"})
    if not next_data_script:
        raise RuntimeError(
            "Could not find the __NEXT_DATA__ script; unable to parse the page."
        )

    # Parse the JSON data
    next_data_str = next_data_script.string
    data = json.loads(next_data_str)

    # Extract the audioAssets section
    try:
        audio_assets = data["props"]["pageProps"]["episode"]["audioAssets"]
    except KeyError:
        # Fallback for potentially different structure or missing data
        audio_assets = (
            data.get("props", {})
            .get("pageProps", {})
            .get("episodeData", {})  # Try another common key for episode details
            .get("audioAssets", [])
        )
        if (
            not audio_assets
        ):  # One more attempt if `episode` was the correct top-level key
            audio_assets = (
                data.get("props", {})
                .get("pageProps", {})
                .get("episode", {})
                .get("audioAssets", [])
            )
    if not audio_assets:
        logging.error(
            f"No audioAssets found in the page data: {json.dumps(data, indent=2)[:1000]}"
        )
        raise RuntimeError("No audioAssets found in the page.")

    m3u8_api_link = None
    for asset in audio_assets:
        if asset.get("format") == "HLS" and "url" in asset:
            m3u8_api_link = asset["url"]
            break

    if not m3u8_api_link:
        raise RuntimeError("No HLS (m3u8) link found in audioAssets.")

    logging.info(f"Found m3u8 API link: {m3u8_api_link}")

    # Follow redirects if needed
    try:
        with requests.get(
            m3u8_api_link, allow_redirects=True, stream=True, timeout=10
        ) as r:
            r.raise_for_status()
            final_m3u8_url = r.url
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to get final m3u8 URL from {m3u8_api_link}: {e}")
        raise

    if not final_m3u8_url.endswith(".m3u8"):
        logging.warning(
            f"Warning: The final URL '{final_m3u8_url}' may not end with .m3u8, but ffmpeg should still try."
        )

    logging.info(f"Final m3u8 download URL: {final_m3u8_url}")

    # Generate a safe title for the MP3 filename
    episode_title = (
        data.get("props", {})
        .get("pageProps", {})
        .get("episode", {})  # Standard path
        .get("title")
    )
    if not episode_title:  # Fallback path
        episode_title = (
            data.get("props", {})
            .get("pageProps", {})
            .get("episodeData", {})
            .get("title", "dr_output")  # Default if still not found
        )

    safe_title = normalize_filename(episode_title)
    mp3_filename = f"{safe_title}.mp3"

    logging.info(f"Downloading to MP3: {mp3_filename}")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output files without asking
        "-protocol_whitelist",
        "file,http,https,tcp,tls,crypto",  # Ensure necessary protocols for HLS
        "-i",
        final_m3u8_url,
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",  # Good quality VBR
        # "-b:a", "256k", # Constant bitrate, -q:a is often preferred for quality/size
        mp3_filename,
    ]
    try:
        subprocess.run(
            ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        logging.info(f"Download and MP3 conversion complete: {mp3_filename}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg download/conversion failed for {final_m3u8_url}:")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Stdout: {e.stdout.decode(errors='ignore')}")
        logging.error(f"Stderr: {e.stderr.decode(errors='ignore')}")
        raise

    # Convert MP3 -> WAV for Whisper processing
    wav_filename = os.path.splitext(mp3_filename)[0] + ".wav"
    convert_audio(mp3_filename, wav_filename)

    return mp3_filename, wav_filename


#################################
# 2) Generate SRT via Whisper-CLI
#################################
def generate_srt_with_whisper(wav_file, model_path, language="da", max_line_len=200):
    """
    Use whisper-cli to generate an SRT subtitle file from a WAV file.
    The output SRT file will have the same base name as the WAV file, but with .srt extension.
    Returns the name of the generated SRT file.
    """
    base_name_for_output = os.path.splitext(wav_file)[0]
    # whisper-cli will append .srt to the -of argument if it's a basename
    srt_file = base_name_for_output + ".srt"

    cmd = [
        "whisper-cli",
        "-m",
        model_path,
        "-f",
        wav_file,
        "-l",
        language,
        "-osrt",  # Output format SRT
        "-ml",
        str(max_line_len),  # Max line length
        "-of",
        base_name_for_output,  # Output file basename
    ]
    logging.info(f"Running whisper-cli to generate SRT: {' '.join(cmd)}")
    try:
        # Capture output for better debugging if needed
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if process.stdout:
            logging.debug(f"Whisper-CLI STDOUT: {process.stdout}")
        if process.stderr:  # whisper-cli often prints progress to stderr
            logging.debug(f"Whisper-CLI STDERR: {process.stderr}")

    except subprocess.CalledProcessError as e:
        logging.error(f"whisper-cli execution failed:")
        logging.error(f"Command: {' '.join(e.cmd)}")
        if e.stdout:
            logging.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logging.error(f"STDERR: {e.stderr}")
        raise

    if not os.path.exists(srt_file):
        # This case should ideally not happen if whisper-cli -of works as expected.
        # Double check if file exists with a slightly different name due to whisper-cli version/behavior.
        # For instance, whisper might append .<lang>.srt.
        potential_srt_file_lang = f"{base_name_for_output}.{language}.srt"
        if os.path.exists(potential_srt_file_lang):
            srt_file = potential_srt_file_lang
            logging.info(f"Found SRT file with language code: {srt_file}")
        else:
            raise FileNotFoundError(
                f"SRT file {srt_file} (or {potential_srt_file_lang}) not found after whisper-cli execution. Command: {' '.join(cmd)}"
            )

    logging.info(f"SRT generated: {srt_file}")
    return srt_file


#################################
# 3) Convert SRT -> LRC
#################################
def srt_time_to_lrc(srt_time):
    """
    Convert SRT timestamp format (HH:MM:SS,MS) to LRC format (MM:SS.xx).
    """
    match = re.match(
        r"(\d+):(\d+):(\d+)[,.](\d+)", srt_time
    )  # Allow . or , for milliseconds
    if match:
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        total_minutes = hours * 60 + minutes
        # LRC uses centiseconds (hundredths of a second)
        centiseconds = (
            milliseconds // 10 if len(str(milliseconds)) == 3 else milliseconds
        )  # if ms is already 2 digits
        if len(str(milliseconds)) == 1:  # e.g. 0:0:1,2 -> 0:0:1.02
            centiseconds = milliseconds * 10
        elif len(str(milliseconds)) == 2:  # e.g. 0:0:1,12 -> 0:0:1.12
            centiseconds = milliseconds
        elif len(str(milliseconds)) == 3:  # e.g. 0:0:1,123 -> 0:0:1.12
            centiseconds = milliseconds // 10

        return f"[{total_minutes:02}:{seconds:02}.{centiseconds:02}]"
    logging.warning(f"Could not parse SRT time: {srt_time}")
    return "[00:00.00]"  # Return a default valid timestamp


def convert_srt_to_lrc(srt_file):
    """
    Convert a .srt file to .lrc format. Returns the generated .lrc filename.
    """
    base_name = os.path.splitext(srt_file)[0]
    lrc_file = base_name + ".lrc"

    logging.info(f"Converting SRT to LRC: {srt_file} -> {lrc_file}")
    with open(srt_file, "r", encoding="utf-8") as srt, open(
        lrc_file, "w", encoding="utf-8"
    ) as lrc:
        lines = srt.readlines()
        lrc_lines = []

        current_block_timestamp = ""
        current_block_text = []

        for line_num, line_content in enumerate(lines):
            line_content = line_content.strip()

            if not line_content:  # Empty line, signifies end of an SRT entry
                if current_block_timestamp and current_block_text:
                    full_text = " ".join(current_block_text).strip()
                    if full_text:  # Only write if there's text
                        lrc_lines.append(f"{current_block_timestamp}{full_text}")
                    current_block_timestamp = ""
                    current_block_text = []
                continue

            if (
                line_content.isdigit()
                and "-->"
                not in lines[line_num + 1 if line_num + 1 < len(lines) else line_num]
            ):
                # This is likely an index number. If a previous block had text, write it.
                if current_block_timestamp and current_block_text:
                    full_text = " ".join(current_block_text).strip()
                    if full_text:
                        lrc_lines.append(f"{current_block_timestamp}{full_text}")
                    current_block_timestamp = (
                        ""  # Reset for the new block starting with index
                    )
                    current_block_text = []
                continue  # Skip processing this index line further

            if "-->" in line_content:  # Timestamp line
                if (
                    current_block_timestamp and current_block_text
                ):  # Should not happen if SRT is well-formed, but safeguard
                    full_text = " ".join(current_block_text).strip()
                    if full_text:
                        lrc_lines.append(f"{current_block_timestamp}{full_text}")
                    current_block_text = []

                start_time_str = line_content.split(" --> ")[0].strip()
                current_block_timestamp = srt_time_to_lrc(start_time_str)

            elif (
                current_block_timestamp
            ):  # This is a text line for the current timestamp
                current_block_text.append(line_content)
            else:
                # Line is not empty, not an index, not a timestamp, and no current timestamp active
                # This could be malformed SRT or header. Log and skip.
                logging.debug(
                    f"Skipping unexpected line in SRT->LRC conversion: '{line_content}'"
                )

        # Write any remaining text buffer for the last subtitle entry
        if current_block_timestamp and current_block_text:
            full_text = " ".join(current_block_text).strip()
            if full_text:
                lrc_lines.append(f"{current_block_timestamp}{full_text}")

        lrc.write("\n".join(lrc_lines))
    logging.info(f"LRC conversion complete: {lrc_file}")
    return lrc_file


#################################
# 4) Embed LRC lyrics into MP3
#################################
def embed_lyrics(mp3_file, lrc_file):
    """
    Embed lyrics from a .lrc file into an MP3 file using the ID3 USLT frame.
    """
    logging.info(f"Embedding LRC into MP3: {lrc_file} -> {mp3_file}")
    if not os.path.exists(mp3_file):
        logging.error(f"MP3 file not found for embedding lyrics: {mp3_file}")
        return
    if not os.path.exists(lrc_file):
        logging.error(f"LRC file not found for embedding: {lrc_file}")
        return

    try:
        audio = MP3(mp3_file, ID3=ID3)
        if audio.tags is None:
            audio.add_tags()

        # Remove existing USLT frames to avoid duplicates, if any
        for key in list(audio.tags.keys()):  # Iterate over a copy of keys
            if key.startswith("USLT"):
                del audio.tags[key]

        with open(lrc_file, "r", encoding="utf-8") as f:
            lyrics = f.read()

        lang_code = "dan"  # Default for original Danish
        # Try to get lang from filename e.g. _english_translated.lrc
        match = re.search(r"_([a-z]{2,})_translated\.lrc$", lrc_file.lower())
        if match:
            lang_name_from_file = match.group(1)
            # rudimentary mapping, extend as needed for 3-letter ISO 639-2 codes
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
                "da": "dan",  # For completeness, though default is dan
            }
            lang_code = lang_map.get(
                lang_name_from_file, "und"
            )  # 'und' for undetermined
        elif "_translated" not in lrc_file and ".lrc" in lrc_file:  # Original
            pass  # lang_code remains 'dan'
        else:  # Translated but couldn't determine lang from filename
            lang_code = "und"

        audio.tags.add(
            USLT(
                encoding=Encoding.UTF8,  # UTF-8 is widely supported
                lang=lang_code,
                desc="Lyrics",  # Description can be customized, e.g., "Translated Lyrics"
                text=lyrics,
            )
        )
        audio.save(v2_version=3)  # Save ID3v2.3 for wider compatibility
        logging.info(
            f"Successfully embedded lyrics into {mp3_file} (lang: {lang_code})"
        )
    except Exception as e:
        logging.error(
            f"Error embedding lyrics into {mp3_file} from {lrc_file}: {e}",
            exc_info=True,
        )


###########################################
# 5) Clean up intermediate files
###########################################
def clean_intermediate_files(files_to_clean, protected_files_abs):
    """
    Delete files listed in the files_to_clean parameter, avoiding protected files.
    """
    for f_to_clean in files_to_clean:
        if f_to_clean and os.path.exists(f_to_clean):
            f_to_clean_abs = os.path.abspath(f_to_clean)
            if f_to_clean_abs in protected_files_abs:
                logging.info(f"Skipping deletion of protected file: {f_to_clean}")
                continue
            try:
                os.remove(f_to_clean)
                logging.info(f"Deleted intermediate file: {f_to_clean}")
            except Exception as e:
                logging.warning(f"Could not delete {f_to_clean}: {e}")
        elif f_to_clean:  # File was in list but doesn't exist
            logging.debug(
                f"Intermediate file listed for deletion not found: {f_to_clean}"
            )


#########################################################
# Main flow: Download -> Whisper -> SRT -> LRC -> Embed -> Cleanup
#########################################################
if __name__ == "__main__":

    protected_files_abs = []  # Absolute paths of files that should not be deleted

    if len(sys.argv) < 2:
        print(
            "Usage: python3 dr_lrc.py <DR_LYD_URL or audio_file_path> [target_language_for_translation]"
        )
        print("Example: python3 dr_lrc.py <URL_OR_PATH> English")
        print("Example: python3 dr_lrc.py my_podcast.mp3 Chinese")
        print(
            "If target_language is omitted, only original Danish lyrics will be embedded."
        )
        print(f"\nEnvironment variables needed:")
        print(
            f"  WHISPER_MODEL_PATH: Path to your Whisper model file (e.g., /models/ggml-medium.bin)"
        )
        print(
            f"  If translating, {GEMINI_API_KEYS_ENV_VAR}: Your Gemini API key(s), comma-separated."
        )
        sys.exit(1)

    model_path = os.environ.get("WHISPER_MODEL_PATH")
    if not model_path or not os.path.exists(model_path):
        logging.error(
            f"Error: WHISPER_MODEL_PATH ('{model_path}') is not set or file does not exist."
        )
        sys.exit(1)
    logging.info(f"Using Whisper model from: {model_path}")

    input_arg = sys.argv[1]
    target_language_for_translation = None

    if len(sys.argv) > 2:
        target_language_for_translation = sys.argv[2]
        logging.info(
            f"Translation requested for target language: {target_language_for_translation}"
        )
        if not GEMINI_API_KEYS:
            logging.error(
                f"Error: Translation requested, but {GEMINI_API_KEYS_ENV_VAR} environment variable is not set or empty."
            )
            sys.exit(1)
        logging.info(
            f"Using Gemini API for translation with {len(GEMINI_API_KEYS)} key(s)."
        )
    else:
        logging.info(
            "No translation target language provided; proceeding with original Danish lyrics only."
        )

    mp3_file_final_original = None  # Path to the MP3 with original lyrics
    wav_file_for_whisper = None

    # These will hold paths to files created by *this script run* that might be cleaned up.
    created_mp3_files_this_run = []
    created_wav_files_this_run = []
    created_srt_files_this_run = []
    created_lrc_files_this_run = []

    if "dr.dk" in input_arg:  # URL Input
        url = input_arg
        logging.info(f"Processing URL: {url}")
        try:
            temp_mp3, temp_wav = download_dr_radio(url)
            mp3_file_final_original = temp_mp3
            wav_file_for_whisper = temp_wav
            created_mp3_files_this_run.append(temp_mp3)
            created_wav_files_this_run.append(temp_wav)
        except Exception as e:
            logging.error(
                f"Failed to download and prepare audio from URL {url}: {e}",
                exc_info=True,
            )
            sys.exit(1)
    else:  # File Input
        if not os.path.exists(input_arg):
            logging.error(f"Error: Input file does not exist: {input_arg}")
            sys.exit(1)

        input_arg_abs = os.path.abspath(input_arg)
        protected_files_abs.append(input_arg_abs)
        logging.info(
            f"Processing local file: {input_arg_abs} (this file will be protected from deletion)"
        )

        input_ext = os.path.splitext(input_arg)[1].lower()
        base_name_input = os.path.splitext(input_arg_abs)[
            0
        ]  # Use abspath for consistent naming

        if input_ext == ".mp3":
            mp3_file_final_original = input_arg_abs
            wav_file_for_whisper = base_name_input + ".wav"
            logging.info(
                f"Input is MP3: {mp3_file_final_original}. Will create WAV: {wav_file_for_whisper}"
            )
            convert_audio(mp3_file_final_original, wav_file_for_whisper)
            created_wav_files_this_run.append(wav_file_for_whisper)
        elif input_ext == ".wav":
            wav_file_for_whisper = input_arg_abs
            mp3_file_final_original = base_name_input + ".mp3"
            logging.info(
                f"Input is WAV: {wav_file_for_whisper}. Will create MP3: {mp3_file_final_original}"
            )
            convert_audio(wav_file_for_whisper, mp3_file_final_original)
            created_mp3_files_this_run.append(mp3_file_final_original)
        else:  # Other audio formats
            logging.info(
                f"Input is other audio format: {input_arg_abs}. Will convert to WAV and MP3."
            )
            try:
                wav_file_for_whisper = base_name_input + ".wav"
                mp3_file_final_original = base_name_input + ".mp3"
                convert_audio(input_arg_abs, wav_file_for_whisper)
                convert_audio(input_arg_abs, mp3_file_final_original)
                created_wav_files_this_run.append(wav_file_for_whisper)
                created_mp3_files_this_run.append(mp3_file_final_original)
            except Exception as e:
                logging.error(
                    f"Error converting input file {input_arg_abs}: {e}", exc_info=True
                )
                sys.exit(1)

    if not mp3_file_final_original or not os.path.exists(mp3_file_final_original):
        logging.error(
            f"MP3 file for processing ({mp3_file_final_original}) not available. Exiting."
        )
        sys.exit(1)
    if not wav_file_for_whisper or not os.path.exists(wav_file_for_whisper):
        logging.error(
            f"WAV file for Whisper ({wav_file_for_whisper}) not available. Exiting."
        )
        sys.exit(1)

    try:
        srt_file = generate_srt_with_whisper(
            wav_file_for_whisper, model_path, language="da", max_line_len=120
        )
        created_srt_files_this_run.append(srt_file)

        lrc_file_original_lang = convert_srt_to_lrc(srt_file)
        created_lrc_files_this_run.append(lrc_file_original_lang)

        embed_lyrics(mp3_file_final_original, lrc_file_original_lang)

        translated_mp3_file_final = None  # Path to the MP3 with translated lyrics

        if target_language_for_translation:
            base_name_lrc = os.path.splitext(lrc_file_original_lang)[0]
            lrc_file_translated = (
                base_name_lrc
                + f"_{target_language_for_translation.lower()}_translated.lrc"
            )
            created_lrc_files_this_run.append(lrc_file_translated)

            logging.info(
                f"Starting translation to {target_language_for_translation} for LRC: {lrc_file_translated}"
            )
            process_lrc_file_batch_dict(
                input_file=lrc_file_original_lang,
                output_file=lrc_file_translated,
                target_language=target_language_for_translation,
                batch_size=50,  # <<< USER CAN MODIFY BATCH SIZE HERE <<<
                debug=os.getenv("DEBUG_TRANSLATION", "False").lower() == "true",
            )

            translated_mp3_file_final = (
                os.path.splitext(mp3_file_final_original)[
                    0
                ]  # Base on original MP3's name
                + f"_{target_language_for_translation.lower()}_translated.mp3"
            )
            # Important: If mp3_file_final_original was the input file, copying it creates a new file.
            # If mp3_file_final_original was *created* by this script, we are copying that created file.
            shutil.copyfile(mp3_file_final_original, translated_mp3_file_final)
            logging.info(
                f"Copied {mp3_file_final_original} to {translated_mp3_file_final} for translated lyrics."
            )
            # This new MP3 is also a product of this run. If it's distinct from original input.
            if os.path.abspath(translated_mp3_file_final) not in protected_files_abs:
                created_mp3_files_this_run.append(translated_mp3_file_final)

            embed_lyrics(translated_mp3_file_final, lrc_file_translated)
            logging.info(
                f"Created MP3 with translated lyrics: {translated_mp3_file_final}"
            )

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}", exc_info=True)
        # Decide if to attempt cleanup or exit immediately. For now, attempt cleanup.
    finally:
        # Consolidate all created files for cleanup, excluding protected ones.
        all_intermediate_files = []
        all_intermediate_files.extend(created_wav_files_this_run)
        all_intermediate_files.extend(created_srt_files_this_run)
        all_intermediate_files.extend(created_lrc_files_this_run)

        # Careful with MP3s: if an MP3 was input, it's protected.
        # If an MP3 was created from WAV/other input, it becomes mp3_file_final_original.
        # If translation occurs, translated_mp3_file_final is also created.
        # We generally want to keep mp3_file_final_original and translated_mp3_file_final.
        # So, created_mp3_files_this_run should typically NOT be cleaned unless they are truly temp.
        # The current logic: mp3_file_final_original is the "main" original output.
        # translated_mp3_file_final is the "main" translated output. These should be kept.
        # Only if created_mp3_files_this_run contains other temp MP3s (not current logic).

        # For now, let's assume WAVs, SRTs, and LRCs are primary cleanup targets.
        # The mp3_file_final_original and translated_mp3_file_final are the desired outputs.

        # Filter out None and duplicates, and ensure they are not the final output MP3s.
        files_to_potentially_remove = list(set(filter(None, all_intermediate_files)))

        # Explicitly keep final output MP3s if they somehow ended up in created lists and aren't protected.
        final_outputs_to_keep_abs = []
        if mp3_file_final_original:
            final_outputs_to_keep_abs.append(os.path.abspath(mp3_file_final_original))
        if (
            "translated_mp3_file_final" in locals() and translated_mp3_file_final
        ):  # check if defined
            final_outputs_to_keep_abs.append(os.path.abspath(translated_mp3_file_final))

        actual_files_to_remove = [
            f
            for f in files_to_potentially_remove
            if os.path.abspath(f) not in final_outputs_to_keep_abs
        ]

        if actual_files_to_remove:
            logging.info(
                f"Attempting to clean up intermediate files: {actual_files_to_remove}"
            )
            clean_intermediate_files(actual_files_to_remove, protected_files_abs)
        else:
            logging.info(
                "No intermediate files marked for cleanup, or all were protected/final outputs."
            )

    logging.info(f"--- Script Execution Finished ---")
    if mp3_file_final_original and os.path.exists(mp3_file_final_original):
        logging.info(
            f"Final output file with original lyrics: {mp3_file_final_original}"
        )
    if (
        "translated_mp3_file_final" in locals()
        and translated_mp3_file_final
        and os.path.exists(translated_mp3_file_final)
    ):
        logging.info(
            f"Additionally created translated file: {translated_mp3_file_final}"
        )
    elif target_language_for_translation and (
        "translated_mp3_file_final" not in locals() or not translated_mp3_file_final
    ):
        logging.warning(
            f"Translation was requested, but the translated MP3 file was not created or its path is missing."
        )
