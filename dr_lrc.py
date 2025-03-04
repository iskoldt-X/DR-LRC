import re
import json
import subprocess
import requests
from bs4 import BeautifulSoup
import os
import sys
import shutil  # For copying mp3 to a new filename

# --- Additional import for translation ---
import ollama

from mutagen.mp3 import MP3
from mutagen.id3 import ID3, USLT, Encoding

#################################
# Translation-related utilities
#################################
def try_parse_json(raw_str):
    """
    Try to parse the string into JSON. If it fails, return None.
    It can also perform pre- and post-cleaning to remove any extra text from the model's output.
    """
    raw_str = raw_str.strip()
    # Sometimes the model outputs something like ```json ... ```, so perform simple stripping.
    raw_str = re.sub(r"^```(json)?|```$", "", raw_str, flags=re.IGNORECASE).strip()

    try:
        data = json.loads(raw_str)
        return data
    except json.JSONDecodeError:
        return None

def batch_translate_dict_mode(danish_texts, debug=False, max_retries=3):
    """
    Wrap multiple lines of Danish into a dict, e.g. {"0": "line0", "1": "line1", ...},
    and require the model to output the corresponding JSON (with the same keys,
    where each value is translated into English).
    """
    # Prepare the dict.
    data_to_translate = { str(i): text for i, text in enumerate(danish_texts) }

    # Construct system prompt.
    system_prompt = (
        "You are a translation model.\n"
        "Your ONLY task is to translate each value in a given JSON dictionary from Danish to English.\n"
        "You MUST return a valid JSON object with the SAME keys, in valid JSON format, and no extra fields.\n"
        "DO NOT include any chain-of-thought or explanations.\n"
        "If a key is '0', output the translation in the value for '0', etc.\n"
        "No commentary. No disclaimers.\n"
        "If you cannot parse something, just do your best. The final response must be valid JSON.\n"
    )
    # Construct user prompt.
    user_prompt = (
        "Translate the following JSON from Danish to English, preserving the keys exactly.\n\n"
        "Input JSON:\n"
        + json.dumps(data_to_translate, ensure_ascii=False, indent=2)
        + "\n\n"
        "Output must be valid JSON with the same keys, each value replaced by the English translation.\n"
    )

    for attempt in range(max_retries):
        response = ollama.chat(
            model='gemma2',  # or the model you prefer
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )

        raw_output = response['message']['content']

        if debug:
            print(f"\n[DEBUG] Attempt {attempt+1} - Model raw output:\n{raw_output}\n")

        # Attempt to parse JSON.
        parsed = try_parse_json(raw_output)
        if parsed is not None and isinstance(parsed, dict):
            # Optionally verify we have each key
            # (here we do a minimal check, but partial success is accepted)
            return parsed
        else:
            if debug:
                print("[DEBUG] JSON parse failed or not a dict - retrying...\n")

    # If all attempts fail, return an empty dict (or you could return None).
    return {}

def process_lrc_file_batch_dict(input_file, output_file, batch_size=10, debug=False):
    """
    Read a .lrc file, translate its content in batches using the "JSON dict" method,
    and insert the translation into a new LRC file as: original / translation.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract all lines that need translation.
    translation_indices = []
    danish_texts = []
    for idx, line in enumerate(lines):
        match = re.match(r"(\[\d{2}:\d{2}\.\d{2}\])(.*)", line.strip())
        if match:
            timestamp, danish_line = match.groups()
            if danish_line.strip():
                translation_indices.append(idx)
                danish_texts.append(danish_line.strip())

    translations = []
    # Translate in batches.
    for i in range(0, len(danish_texts), batch_size):
        batch = danish_texts[i:i+batch_size]
        if debug:
            print(f"\n[DEBUG] Processing batch {i // batch_size + 1}: {batch}")

        result_dict = batch_translate_dict_mode(batch, debug=debug)
        # Extract translations in order.
        for j in range(len(batch)):
            key_str = str(j)
            translation_value = result_dict.get(key_str, "")
            translations.append(translation_value.strip())

    # Insert the translation results into the file lines.
    translated_lines = lines.copy()
    for idx, translation in zip(translation_indices, translations):
        match = re.match(r"(\[\d{2}:\d{2}\.\d{2}\])(.*)", lines[idx].strip())
        if match:
            timestamp, original_text = match.groups()
            # Combine original text and translation
            translated_lines[idx] = f"{timestamp}{original_text} / {translation}\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(translated_lines)

    print(f"✅ Translation completed. Output saved to: {output_file}")

#########################################################
# 1) Download from DR LYD -> MP3, then convert MP3 -> WAV
#########################################################
def download_dr_radio(url):
    """
    Given a full DR LYD URL, extract the m3u8 link from the page, use ffmpeg to download/convert to MP3,
    then create a WAV version for Whisper processing. Returns (mp3_filename, wav_filename).
    """
    # Request the page
    resp = requests.get(url)
    resp.raise_for_status()
    html = resp.text

    # Use BeautifulSoup to find the <script id="__NEXT_DATA__"> block
    soup = BeautifulSoup(html, "html.parser")
    next_data_script = soup.find("script", attrs={"id": "__NEXT_DATA__"})
    if not next_data_script:
        raise RuntimeError("Could not find the __NEXT_DATA__ script; unable to parse the page.")

    # Parse the JSON data
    next_data_str = next_data_script.string
    data = json.loads(next_data_str)

    # Extract the audioAssets section
    audio_assets = (
        data.get("props", {})
            .get("pageProps", {})
            .get("episode", {})
            .get("audioAssets", [])
    )
    if not audio_assets:
        raise RuntimeError("No audioAssets found in the page.")

    m3u8_api_link = None
    for asset in audio_assets:
        if asset.get("format") == "HLS" and "url" in asset:
            m3u8_api_link = asset["url"]
            break

    if not m3u8_api_link:
        raise RuntimeError("No HLS (m3u8) link found in audioAssets.")

    print(f"Found m3u8 API link: {m3u8_api_link}")

    # Follow redirects if needed
    with requests.get(m3u8_api_link, allow_redirects=True, stream=True) as r:
        r.raise_for_status()
        final_m3u8_url = r.url

    if not final_m3u8_url.endswith(".m3u8"):
        print("Warning: The final URL may not end with .m3u8, but ffmpeg should still work.")

    print(f"Final m3u8 download URL: {final_m3u8_url}")

    # Generate a safe title for the MP3 filename
    episode_title = (
        data.get("props", {})
            .get("pageProps", {})
            .get("episode", {})
            .get("title", "dr_output")
    )
    safe_title = re.sub(r"[^\w\u4e00-\u9fa5-]+", "_", episode_title)
    mp3_filename = f"{safe_title}.mp3"

    print(f"Downloading to MP3: {mp3_filename}")
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", final_m3u8_url,
        "-c:a", "libmp3lame",
        "-b:a", "256k",
        mp3_filename
    ]
    subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Download and MP3 conversion complete: {mp3_filename}")

    # Convert MP3 -> WAV for Whisper processing
    wav_filename = f"{safe_title}.wav"
    ffmpeg_cmd_wav = [
        "ffmpeg",
        "-y",
        "-i", mp3_filename,
        "-c:a", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        wav_filename
    ]
    subprocess.run(ffmpeg_cmd_wav, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"WAV conversion complete: {wav_filename}")

    return mp3_filename, wav_filename

#################################
# 2) Generate SRT via Whisper-CLI
#################################
def generate_srt_with_whisper(wav_file, model_path, language="da", max_line_len=100):
    """
    Use whisper-cli to generate an SRT subtitle file from a WAV file.
    The output SRT file will have the same base name as the WAV file, but with .srt extension.
    Returns the name of the generated SRT file.
    """
    base_name = os.path.splitext(wav_file)[0]
    srt_file = base_name + ".srt"

    cmd = [
        "whisper-cli",
        "-m", model_path,
        "-f", wav_file,
        "-l", language,
        "-osrt",
        "-ml", str(max_line_len),
        "-of", base_name
    ]
    print(f"Running whisper-cli to generate SRT: {cmd}")
    subprocess.run(cmd, check=True)
    print(f"SRT generated: {srt_file}")
    return srt_file

#################################
# 3) Convert SRT -> LRC
#################################
def srt_time_to_lrc(srt_time):
    """
    Convert SRT timestamp format (HH:MM:SS,MS) to LRC format (MM:SS.xx).
    """
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", srt_time)
    if match:
        hours, minutes, seconds, milliseconds = map(int, match.groups())
        total_minutes = hours * 60 + minutes
        return f"[{total_minutes:02}:{seconds:02}.{milliseconds//10:02}]"
    return ""

def convert_srt_to_lrc(srt_file):
    """
    Convert a .srt file to .lrc format. Returns the generated .lrc filename.
    """
    base_name = os.path.splitext(srt_file)[0]
    lrc_file = base_name + ".lrc"

    print(f"Converting SRT to LRC: {srt_file} -> {lrc_file}")
    with open(srt_file, "r", encoding="utf-8") as srt, open(lrc_file, "w", encoding="utf-8") as lrc:
        lines = srt.readlines()
        lrc_lines = []
        timestamp = ""
        for line in lines:
            line = line.strip()
            if "-->" in line:  # Timestamp line
                start_time = line.split(" --> ")[0]
                timestamp = srt_time_to_lrc(start_time)
            elif line and not line.isdigit():  # Subtitle text (non-blank, not just the index)
                lrc_lines.append(f"{timestamp}{line}")
        lrc.write("\n".join(lrc_lines))
    print(f"LRC conversion complete: {lrc_file}")
    return lrc_file

#################################
# 4) Embed LRC lyrics into MP3
#################################
def embed_lyrics(mp3_file, lrc_file):
    """
    Embed lyrics from a .lrc file into an MP3 file using the ID3 USLT frame.
    """
    print(f"Embedding LRC into MP3: {lrc_file} -> {mp3_file}")
    try:
        audio = MP3(mp3_file, ID3=ID3)
        if audio.tags is None:
            audio.add_tags()
        with open(lrc_file, "r", encoding="utf-8") as f:
            lyrics = f.read()
        audio.tags.add(
            USLT(
                encoding=Encoding.UTF8,
                lang='dan',
                desc='Lyrics',
                text=lyrics
            )
        )
        audio.save()
        print(f"Successfully embedded lyrics into {mp3_file}")
    except Exception as e:
        print(f"Error embedding lyrics: {e}")

###########################################
# 5) Clean up intermediate files
###########################################
def clean_intermediate_files(files):
    """
    Delete files listed in the files parameter.
    """
    for f in files:
        try:
            os.remove(f)
            print(f"Deleted intermediate file: {f}")
        except Exception as e:
            print(f"Could not delete {f}: {e}")

#########################################################
# Main flow: Download -> Whisper -> SRT -> LRC -> Embed -> Cleanup
#########################################################
if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python3 dr_lrc.py <DR_LYD_URL> [translate]")
        sys.exit(1)

    url = sys.argv[1]
    do_translate = (len(sys.argv) > 2 and sys.argv[2].lower() == "translate")

    print(f"Processing URL: {url}")
    print(f"Translation mode: {do_translate}")

    # Get the model path from environment variable WHISPER_MODEL_PATH
    model_path = os.environ.get("WHISPER_MODEL_PATH")
    if not model_path:
        print("Error: WHISPER_MODEL_PATH environment variable is not set.")
        sys.exit(1)
    print(f"Using Whisper model from: {model_path}")

    # Step 1: Download -> MP3 and produce WAV
    mp3_file, wav_file = download_dr_radio(url)

    # Step 2: Generate SRT using whisper-cli
    srt_file = generate_srt_with_whisper(wav_file, model_path, language="da", max_line_len=100)

    # Step 3: Convert SRT -> LRC
    lrc_file = convert_srt_to_lrc(srt_file)

    # Step 4: Embed the LRC into the MP3 (original Danish)
    embed_lyrics(mp3_file, lrc_file)

    # --- New Step: If "translate" requested, produce a second MP3 with translated lyrics ---
    if do_translate:
        base_name = os.path.splitext(lrc_file)[0]
        translated_lrc_file = base_name + "_translated.lrc"

        # Translate and produce a new LRC (Danish / English)
        process_lrc_file_batch_dict(
            input_file=lrc_file,
            output_file=translated_lrc_file,
            batch_size=10,
            debug=True
        )

        # Make a second MP3 file, e.g. "safe_title_translated.mp3"
        translated_mp3_file = os.path.splitext(mp3_file)[0] + "_translated.mp3"
        shutil.copyfile(mp3_file, translated_mp3_file)

        # Embed the new LRC (with translations) into the second MP3
        embed_lyrics(translated_mp3_file, translated_lrc_file)

        print(f"✅ Created translated MP3: {translated_mp3_file}")

    # Step 5: Clean up intermediate files (WAV, SRT, LRC)
    # Decide whether you also want to remove the translated LRC if do_translate == True
    files_to_remove = [wav_file, srt_file, lrc_file]
    # If you want to remove the translated LRC as well, uncomment:
    # if do_translate:
    #     files_to_remove.append(translated_lrc_file)

    clean_intermediate_files(files_to_remove)

    print("All steps completed. Final output file:", mp3_file)
    if do_translate:
        print("Additionally created translated file:", translated_mp3_file)
