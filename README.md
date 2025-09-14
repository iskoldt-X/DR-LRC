# DR-LRC

A Python script to download **DR LYD** audio, generate subtitles using **Whisper**, convert the subtitles to LRC lyrics format, optionally translate them into any language via **Ollama** + **Gemma 3**, and embed these lyrics into the MP3 file.

**Now also supports local audio files (MP3 or WAV) for transcription, translation, and lyric embedding – no DR link required.**


> **Disclaimer:** This project is intended for educational and research purposes only. The author does not endorse or promote the unauthorized distribution of copyrighted content. Please respect the rights of content creators and only use this tool to download content that is freely available on the internet and use it for study Danish and only for personal use. The code was written by the help of ChatGPT.


## Features

- **Download DR LYD audio:** Scrapes a DR LYD program page to extract the m3u8 URL and downloads the audio as an MP3 file using ffmpeg.
- **Convert audio for Whisper:** Converts the downloaded MP3 file to WAV (mono, 16 kHz) for compatibility with Whisper.
- **(NEW) Transcribe local audio files:** You can pass a local MP3 or WAV file to the script. It will convert, transcribe, and optionally translate it just like a DR podcast.
- **Generate subtitles:** Uses Whisper CLI to generate an SRT subtitle file from the WAV file.
- **Convert SRT to LRC:** Converts the generated SRT file into LRC (lyric) format.
- **Embed lyrics:** Embeds the LRC lyrics into the MP3 file using ID3 tags.
- **(Optional) Generate a translated copy:** If you pass a second argument (either `translate` or a specific language name), a second MP3 is created containing Danish + translation.
- **Cleanup:** Removes intermediate files (WAV, SRT, LRC) so that only the final MP3 remains.

- **(Optional) Generate a translated copy:** If you pass a second argument (either `translate` or a specific language name), a second MP3 is created containing Danish + translation.
- **Cleanup:** Removes intermediate files (WAV, SRT, LRC) so that only the final MP3 remains.

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) must be installed and in your system PATH.
- [whisper-cli](https://github.com/ggerganov/whisper.cpp) must be installed. You can see the installation instructions for each platform in the repository.
- `./build/bin/whisper-cli` must be in your system PATH.
- Environment variable `WHISPER_MODEL_PATH` must be set to the path of your Whisper model file (e.g., `ggml-medium.bin`).
- (Optional) Translation: If you plan to use the translate feature, you need Ollama and the Python ollama client installed, as well as an available model (e.g. `gemma3`).

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/iskoldt-X/DR-LRC.git
   cd DR-LRC
    ```


2. **Install the required Python packages:**

   ```bash
   pip install -r requirements.txt
    ```
3. **Set up the Whisper model path in your shell.**

   For example, if you are using zsh on macOS


   
   A quick guide to install `whisper-cli` and the model:

   ```bash

   git clone https://github.com/ggml-org/whisper.cpp.git
   cd whisper.cpp

   # using large-v3 model for example
   make -j large-v3     
    ```

    After that, you have your model in `./models/ggml-large-v3.bin` and `whisper-cli` in `./build/bin/whisper-cli`.

    Add the following line to your `~/.zshrc` file:

   ```bash
   export WHISPER_MODEL_PATH=/path/to/your/whisper.cpp/models/ggml-large-v3.bin
   export PATH="/path/to/your/whisper.cpp/build/bin:$PATH"
   ```
   Replace `/path/to/your/` with the actual path to your `whisper.cpp` directory.

   Then, run `source ~/.zshrc` to apply the changes.

    Download and ollama and then pull the gemma3 model in the terminal:
    ```bash
    ollama pull gemma3:12b
    ```
    You can also use other models like `gemma3:4b` or `gemma3:1b` if you don't have enough GPU memory. Or you have so much GPU memory that you can use `gemma3:27b`. Just remember to change the model name in the script.


## Usage

### 1. Process DR LYD Podcast

Run the script from the command line by providing a DR LYD URL:

```bash
python3 dr_lrc.py https://www.dr.dk/lyd/your-podcast-link
```

The script will:

1. Download the audio and convert it to MP3.
2. Convert the MP3 to WAV for transcription.
3. Run whisper-cli to generate an SRT subtitle file.
4. Convert the SRT file to LRC lyrics.
5. Embed the LRC lyrics into the MP3.
6. Delete intermediate files, leaving only the final MP3 file.

### 2. Process Local Audio File

You can also use the script to process a local audio file (MP3 or WAV). Just provide the path to the file as the first argument:

```bash
python3 dr_lrc.py your_file.mp3
```
This will transcribe the audio, generate Danish subtitles and lyrics, and embed them into a new MP3 file. If the input is a WAV file, it will be converted to MP3 automatically.

### Translation
You can optionally add a second argument to request translation into a specific language. For example, if you run:



```bash
python3 dr_lrc.py <DR_LYD_URL> French
```

It will create a second MP3 containing Danish + French lyrics embedded. The main MP3 remains Danish-only.

If you include the word translate as the second argument, the script will default to English. For instance:
    
```bash
python3 dr_lrc.py <DR_LYD_URL> translate
```

You will end up with two MP3s:

The original MP3 (with Danish-only lyrics).
A second MP3 named something like dr_output_translated.mp3, containing “Danish / English” lines.



## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgements
- [DR LYD](https://www.dr.dk/lyd)
- [Whisper](https://github.com/openai/whisper)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp)
- [ffmpeg](https://ffmpeg.org/)
- [mutagen](https://github.com/quodlibet/mutagen)
- [Ollama](https://ollama.com)
- [Gemma 3](https://ollama.com/library/gemma3:12b)

