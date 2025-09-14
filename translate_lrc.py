#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LRC File Translator using Gemini API (Corrected to match original logic)

This script translates the lyrical content of an LRC file from Danish to a specified target language.
It uses the exact same Gemini API logic as the original dr_lrc_gemini.py script.

Usage:
    python3 translate_lrc.py <path_to_lrc_file> <TargetLanguage> [--debug]

Example:
    python3 translate_lrc.py "my_podcast.lrc" "English"

Dependencies:
    - Python 3.6+
    - Python library: google-generativeai
    - Environment Variable: GEMINI_API_KEYS must be set to your comma-separated Gemini API key(s).
"""

import json
import logging
import os
import random
import re
import sys
import time

try:
    from google import genai
    from google.genai import types
    from google.api_core.exceptions import ResourceExhausted, TooManyRequests
except ImportError:
    print("Error: The 'google-generativeai' library is not installed.")
    print("Please install it using: pip install google-generativeai")
    sys.exit(1)

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# ────────────── Gemini API Config (Strictly following original script) ──────────────
GEMINI_API_KEYS_ENV_VAR = "GEMINI_API_KEYS"
GEMINI_API_KEYS = [
    k.strip() for k in os.getenv(GEMINI_API_KEYS_ENV_VAR, "").split(",") if k.strip()
]

# Client initialization logic, exactly as in the original script
_client_gemini = None
if GEMINI_API_KEYS:
    _current_key_index_gemini = 0
    _request_count_gemini = 0
    # Use genai.Client, not GenerativeModel
    _client_gemini = genai.Client(api_key=GEMINI_API_KEYS[_current_key_index_gemini])
else:
    pass

MAX_PER_API_GEMINI = int(os.getenv("MAX_PER_API_GEMINI", "50"))
MAX_RETRIES_GEMINI = 20
BASE_DELAY_GEMINI = 15

# --- Gemini API Helper Functions (Strictly following original script) ---


def _get_client_gemini() -> genai.Client:
    """Return a Gemini client bound to the current key; rotate when quota exhausted."""
    global _current_key_index_gemini, _request_count_gemini, _client_gemini
    if not GEMINI_API_KEYS:
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
    """Increment the usage counter after each successful completion for Gemini."""
    global _request_count_gemini
    _request_count_gemini += 1


def try_parse_json(raw_str: str):
    """Safely parses a string into JSON, stripping markdown formatting."""
    if not raw_str:
        return None
    raw_str = raw_str.strip()
    raw_str = re.sub(r"^```(json)?|```$", "", raw_str, flags=re.IGNORECASE).strip()
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        return None


def _build_gemini_response_schema_for_dict(keys_list: list[str]) -> dict:
    properties = {key: {"type": "string"} for key in keys_list}
    return {"type": "object", "properties": properties, "required": keys_list}


def batch_translate_dict_mode_gemini(
    danish_texts: list, target_language="English", debug=False
):
    """Translates a batch of texts using the original script's exact API call logic."""
    if not _client_gemini:
        logging.error(
            f"Gemini API client not initialized. Please set {GEMINI_API_KEYS_ENV_VAR}."
        )
        return {}

    data_to_translate = {str(i): text for i, text in enumerate(danish_texts)}
    expected_keys = list(data_to_translate.keys())

    system_prompt = f"""
        You are an expert translation model, specifically for translating Danish podcast subtitles into {target_language}.
        Your ONLY task is to translate the values of the given JSON dictionary from Danish into fluent and contextually appropriate {target_language}.
        
        IMPORTANT CONTEXT RULE: The numbered sentences in the input JSON are sequential parts of a single conversation. A sentence with a higher number comes directly after a sentence with a lower number.
        Pay close attention to sentences that seem incomplete. They are likely continued in the next numbered entry. Your translation should reflect this continuity. For example, if sentence "10" completes the thought of sentence "9", your translation for "10" should flow naturally from your translation for "9", avoiding repetition.

        - Return ONLY a valid JSON object with the exact same keys: {', '.join(f'"{k}"' for k in expected_keys)}.
        - Do NOT add extra fields or explanations.
    """
    user_prompt = f"""
        Translate the values in the following JSON dictionary from Danish into {target_language}.
        Preserve the keys exactly and return only the translated JSON object.
        Input JSON:
        {json.dumps(data_to_translate, ensure_ascii=False, indent=2)}
        Provide the translated JSON ONLY:
    """

    gemini_schema = _build_gemini_response_schema_for_dict(expected_keys)

    last_feedback = None
    for attempt in range(MAX_RETRIES_GEMINI):
        current_user_prompt = user_prompt
        if last_feedback:
            current_user_prompt += (
                f"\n\nPREVIOUS ATTEMPT FAILED: {last_feedback}. PLEASE CORRECT."
            )

        client = _get_client_gemini()  # Use the client object
        raw_output = None
        try:
            # Use client.models.generate_content, the original method
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # model is specified here
                contents=[current_user_prompt],
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=gemini_schema,
                ),
            )
            raw_output = response.text
            _count_request_gemini()

        except (ResourceExhausted, TooManyRequests) as e:
            logging.warning(
                f"Gemini API limit error (attempt {attempt + 1}): {e}. Switching key..."
            )
            _request_count_gemini = MAX_PER_API_GEMINI
            time.sleep(BASE_DELAY_GEMINI * (attempt + 1))
            last_feedback = f"API limit error: {e}"
            continue
        except Exception as e:
            logging.error(
                f"Gemini API request failed (attempt {attempt + 1}): {type(e).__name__} - {e}"
            )
            time.sleep(BASE_DELAY_GEMINI * (attempt + 1))
            last_feedback = f"General API error: {e}"
            continue

        if debug:
            logging.info(
                f"\n[DEBUG] Attempt {attempt+1} - Model raw output:\n{raw_output}\n"
            )

        parsed = try_parse_json(raw_output)
        if (
            parsed
            and isinstance(parsed, dict)
            and set(parsed.keys()) == set(data_to_translate.keys())
        ):
            return parsed
        else:
            last_feedback = f"Output was not valid JSON or keys mismatched. Raw output: '{str(raw_output)[:200]}...'"

    logging.error(f"Failed to translate batch after {MAX_RETRIES_GEMINI} attempts.")
    return {}


# --- Main Script Logic (remains the same) ---


def run_translate_lrc(
    input_file: str,
    target_language: str,
    batch_size: int = 50,
    debug: bool = False,
) -> str | None:
    """
    Translates an LRC file and saves it as a new bilingual LRC file.

    Args:
        input_file: Path to the source .lrc file.
        target_language: The language to translate to (e.g., "English").
        batch_size: The number of lines to translate in each API call.
        debug: If True, enables debug logging for translation.

    Returns:
        The path to the generated translated .lrc file on success, otherwise None.
    """
    logging.info(f"Reading source LRC file: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    original_texts, line_indices = [], []
    for idx, line in enumerate(lines):
        match = re.match(r"(\[.+?\])(.*)", line.strip())
        if match and match.group(2).strip():
            original_texts.append(match.group(2).strip())
            line_indices.append(idx)

    if not original_texts:
        logging.warning("No text found to translate.")
        return None

    base_name = os.path.splitext(input_file)[0]
    output_file = f"{base_name}_{target_language.lower()}_translated.lrc"

    all_translations = {}
    logging.info(
        f"Found {len(original_texts)} lines to translate. Processing in batches of {batch_size}..."
    )

    for i in range(0, len(original_texts), batch_size):
        batch_texts = original_texts[i : i + batch_size]
        logging.info(f"  - Translating batch {i // batch_size + 1}...")
        result_dict = batch_translate_dict_mode_gemini(
            batch_texts, target_language, debug
        )
        if not result_dict:  # Handle API failure for a batch
            logging.error(
                f"Failed to translate batch starting with: '{batch_texts[0]}'. Aborting."
            )
            return None
        for j, text in enumerate(batch_texts):
            all_translations[i + j] = result_dict.get(
                str(j), "Error: Translation missing"
            )

    new_lrc_lines = lines[:]
    for text_idx, translation in all_translations.items():
        original_line_idx = line_indices[text_idx]
        match = re.match(r"(\[.+?\])(.*)", lines[original_line_idx].strip())
        if match:
            timestamp, original_text = match.groups()
            new_lrc_lines[original_line_idx] = (
                f"{timestamp}{original_text.strip()} / {translation.strip()}\n"
            )

    logging.info(f"Writing translated LRC file to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(new_lrc_lines)
    return output_file


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_lrc_path, target_language = sys.argv[1], sys.argv[2]
    debug_mode = "--debug" in sys.argv

    if not os.path.exists(input_lrc_path):
        logging.error(f"Input LRC file not found: '{input_lrc_path}'")
        sys.exit(1)

    if not GEMINI_API_KEYS:
        logging.error(f"Environment variable '{GEMINI_API_KEYS_ENV_VAR}' is not set.")
        sys.exit(1)

    output_lrc_path = run_translate_lrc(
        input_file=input_lrc_path,
        target_language=target_language,
        debug=debug_mode,
    )

    if output_lrc_path:
        print("\n" + "=" * 50)
        logging.info("✅ Translation process completed successfully!")
        logging.info(f"  - Bilingual LRC file saved as: {output_lrc_path}")
        print("=" * 50)
    else:
        logging.error(f"\n❌ An unrecoverable error occurred.")
        sys.exit(1)


if __name__ == "__main__":
    main()
