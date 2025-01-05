import re
import os
import io
import math
import requests

import streamlit as st
import PyPDF2
from gtts import gTTS

# Anthropic (Claude) support
try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# OpenAI GPT
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Hugging Face
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# We'll define a flag to indicate if Ollama is "allowed" or installed.
# We'll assume we want to call its REST endpoint (ollama serve).
OLLAMA_AVAILABLE = True

# --------------------------------------------------------------------------
# Sanitize / Clean text to avoid UnicodeEncodeError with latin-1
# --------------------------------------------------------------------------
def sanitize_text(text: str) -> str:
    """
    Replace or remove certain Unicode “smart” characters
    that can’t be encoded in latin-1 when sending to some APIs.
    """
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2013": "-",
        "\u2014": "-"
    }
    for bad_char, replacement in replacements.items():
        text = text.replace(bad_char, replacement)
    return text

# --------------------------------------------------------------------------
# Summarization with Ollama (via REST endpoint)
# --------------------------------------------------------------------------
def summarize_text_ollama(text, model="llama2", max_tokens=1024, endpoint="http://localhost:11434"):
    """
    Summarize text using Ollama's REST API at the specified endpoint (default: http://localhost:11434).
    
    Requirements:
      - Ollama installed and running in server mode: `ollama serve`
      - A local model (e.g. "llama2") available: `ollama pull llama2`
      - Mac or Linux environment (experimental on Linux).
    """
    if not OLLAMA_AVAILABLE:
        return text  # fallback if Ollama support is disabled

    # Clean up text
    text = sanitize_text(text)

    prompt = f"""
You are a helpful assistant. Please provide a concise summary of the following text:
{text}
"""

    try:
        data = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
        }
        r = requests.post(f"{endpoint}/generate", json=data)
        if r.status_code != 200:
            return f"[OLLAMA ERROR: HTTP {r.status_code}] {text[:200]}"
        return r.text.strip()
    except Exception as e:
        return f"[OLLAMA ERROR: {e}] {text[:200]}"

# --------------------------------------------------------------------------
# Summarization Functions
# --------------------------------------------------------------------------
def summarize_text_claude(text, anthropic_api_key, max_tokens_to_sample=3000, model="claude-2"):
    if not CLAUDE_AVAILABLE:
        return text

    import anthropic
    text = sanitize_text(text)

    client = anthropic.Client(api_key=anthropic_api_key)
    PROMPT_INSTRUCTIONS = """You are a helpful assistant that summarizes text.
Please provide a concise, clear summary of the following excerpt:
"""

    response = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT}{PROMPT_INSTRUCTIONS}{text}{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model=model,
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=0.7
    )
    
    summary = response["completion"].strip()
    return summary

def summarize_text_openai(text, openai_api_key, model="gpt-3.5-turbo", max_tokens=1024):
    if not OPENAI_AVAILABLE:
        return text

    import openai
    openai.api_key = openai_api_key
    text = sanitize_text(text)
    prompt = f"""
You are a helpful assistant. Please summarize the following text clearly and concisely:
{text}
"""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a creative and helpful writing assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def summarize_text_huggingface(text, model_name="facebook/bart-large-cnn", max_length=200):
    if not HF_AVAILABLE:
        return text

    from transformers import pipeline
    text = sanitize_text(text)
    summarizer = pipeline("summarization", model=model_name)
    summary_list = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
    summary = summary_list[0]['summary_text']
    return summary.strip()

# --------------------------------------------------------------------------
# Chunking & Summaries
# --------------------------------------------------------------------------
def chunk_text(text, chunk_size=3000, overlap=300):
    text = text.strip()
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        if start < 0:
            start = 0
    return chunks

def summarize_in_chunks(text, method="Claude", anthropic_api_key="", openai_api_key="",
                        huggingface_model="facebook/bart-large-cnn", ollama_model="llama2",
                        ollama_endpoint="http://localhost:11434",
                        chunk_size=3000, overlap=300):
    """
    1) chunk the text
    2) summarize each chunk
    3) combine partial summaries
    4) optional second pass (chain-of-summaries)
    """
    text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    partial_summaries = []
    
    for chunk in text_chunks:
        if method == "Claude" and anthropic_api_key and CLAUDE_AVAILABLE:
            try:
                partial_summary = summarize_text_claude(chunk, anthropic_api_key)
            except Exception as e:
                partial_summary = f"[CLAUDE ERROR: {e}] {chunk[:200]}"
        elif method == "OpenAI GPT" and openai_api_key and OPENAI_AVAILABLE:
            try:
                partial_summary = summarize_text_openai(chunk, openai_api_key)
            except Exception as e:
                partial_summary = f"[OPENAI ERROR: {e}] {chunk[:200]}"
        elif method == "Hugging Face" and HF_AVAILABLE:
            try:
                partial_summary = summarize_text_huggingface(chunk, huggingface_model)
            except Exception as e:
                partial_summary = f"[HF ERROR: {e}] {chunk[:200]}"
        elif method == "Ollama" and OLLAMA_AVAILABLE:
            try:
                partial_summary = summarize_text_ollama(chunk, model=ollama_model, endpoint=ollama_endpoint)
            except Exception as e:
                partial_summary = f"[OLLAMA ERROR: {e}] {chunk[:200]}"
        else:
            partial_summary = chunk
        
        partial_summaries.append(partial_summary)
    
    combined_summary_text = "\n\n".join(partial_summaries)

    # Chain-of-summaries pass
    if method in ["Claude", "OpenAI GPT", "Hugging Face", "Ollama"]:
        if method == "Claude" and anthropic_api_key and CLAUDE_AVAILABLE:
            final_summary = summarize_text_claude(combined_summary_text, anthropic_api_key)
        elif method == "OpenAI GPT" and openai_api_key and OPENAI_AVAILABLE:
            final_summary = summarize_text_openai(combined_summary_text, openai_api_key)
        elif method == "Hugging Face" and HF_AVAILABLE:
            final_summary = summarize_text_huggingface(combined_summary_text, huggingface_model)
        elif method == "Ollama" and OLLAMA_AVAILABLE:
            final_summary = summarize_text_ollama(combined_summary_text, model=ollama_model, endpoint=ollama_endpoint)
        else:
            final_summary = combined_summary_text
    else:
        final_summary = combined_summary_text
    
    return final_summary

# --------------------------------------------------------------------------
# PDF -> Text -> Chapters -> Summaries -> TTS
# --------------------------------------------------------------------------
def extract_text_from_pdf(file_data):
    text_content = []
    reader = PyPDF2.PdfReader(file_data)
    for page_num in range(len(reader.pages)):
        page_text = reader.pages[page_num].extract_text()
        if page_text:
            text_content.append(page_text)
    return "\n".join(text_content)

def split_into_chapters(text, chapter_pattern=r"CHAPTER\s+\d+"):
    matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))
    if not matches:
        return [("Full_Book", text)]

    chapters = []
    matches.append(
        type('dummy', (object,), {
            'start': lambda group=0: len(text),
            'group': lambda group=0: "END_OF_BOOK"
        })()
    )
    for i in range(len(matches) - 1):
        start_index = matches[i].start()
        end_index = matches[i+1].start()
        chapter_title = matches[i].group(0)
        chapter_text = text[start_index:end_index].strip()
        chapters.append((chapter_title, chapter_text))
    return chapters

def text_to_speech(text, language='en'):
    tts = gTTS(text=text, lang=language)
    mp3_data = io.BytesIO()
    tts.write_to_fp(mp3_data)
    mp3_data.seek(0)
    return mp3_data

# --------------------------------------------------------------------------
# Streamlit App
# --------------------------------------------------------------------------
def main():
    st.title("PDF 2 Podcast")
    st.subheader("Convert your favourite book to a podcast/audiobook!")

    # 1) Choose Audio Type: "Audiobook" or "Podcast"
    audio_type = st.sidebar.selectbox(
        "Choose Output Format",
        ["Podcast", "Audiobook"]
    )

    # If "Podcast," user can pick a Summarization Method
    summarization_method = None
    anthropic_api_key = ""
    openai_api_key = ""
    huggingface_model = "facebook/bart-large-cnn"
    ollama_model = "llama2"
    ollama_endpoint = "http://localhost:11434"

    if audio_type == "Podcast":
        # Summarization method only appears if user selects "Podcast"
        st.sidebar.header("Summarization Method")
        summarization_method = st.sidebar.selectbox(
            "Summarization Engine",
            ["Claude", "OpenAI GPT", "Hugging Face", "Ollama"]
        )

        if summarization_method == "Claude":
            anthropic_api_key = st.sidebar.text_input("Anthropic API Key", type="password")
        elif summarization_method == "OpenAI GPT":
            openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        elif summarization_method == "Hugging Face":
            huggingface_model = st.sidebar.text_input("Hugging Face Model", value="facebook/bart-large-cnn")
        elif summarization_method == "Ollama":
            ollama_model = st.sidebar.text_input("Ollama Model", value="llama2")
            ollama_endpoint = st.sidebar.text_input("Ollama Endpoint", value="http://localhost:11434")

    st.sidebar.header("Chapter Detection")
    chapter_pattern = st.sidebar.text_input("Chapter Regex Pattern", value=r"CHAPTER\s+\d+")

    st.sidebar.header("Chunking Parameters (Podcast Only)")
    chunk_size = st.sidebar.number_input("Chunk size (chars)", value=3000, step=500)
    overlap = st.sidebar.number_input("Overlap (chars)", value=300, step=50)

    st.write(f"""
    **Audio Type:** {audio_type}

    - **Audiobook**: Reads the full PDF text as-is.
    - **Podcast**: Summarizes each chapter with your chosen method, then converts the summary to audio.
    """)

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        if st.button("Process PDF"):
            with st.spinner("Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)

            with st.spinner("Splitting text into chapters..."):
                chapters = split_into_chapters(pdf_text, chapter_pattern)
                st.success(f"Found {len(chapters)} chapter(s).")

            for idx, (ch_title, ch_text) in enumerate(chapters, start=1):
                st.subheader(f"Chapter {idx}: {ch_title}")

                final_text = ch_text  # default to full text (Audiobook)
                
                if audio_type == "Podcast" and summarization_method:
                    # Summarize the text with the chosen method
                    with st.spinner(f"Summarizing chapter {idx} via {summarization_method}..."):
                        final_text = summarize_in_chunks(
                            ch_text,
                            method=summarization_method,
                            anthropic_api_key=anthropic_api_key,
                            openai_api_key=openai_api_key,
                            huggingface_model=huggingface_model,
                            ollama_model=ollama_model,
                            ollama_endpoint=ollama_endpoint,
                            chunk_size=chunk_size,
                            overlap=overlap
                        )

                    # Display partial summary for user
                    st.write("**Podcast Summary:**")
                    st.write(final_text[:2000] + ("..." if len(final_text) > 2000 else ""))
                else:
                    st.write("**Audiobook Text:** (full chapter)")
                    st.write(ch_text[:2000] + ("..." if len(ch_text) > 2000 else ""))

                # Convert final_text to audio
                with st.spinner(f"Generating TTS for chapter {idx}..."):
                    audio_data = text_to_speech(final_text)

                safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", ch_title)
                filename = f"{idx}_{safe_title}.mp3"

                # Streamlit audio + download
                st.audio(audio_data, format="audio/mp3")
                st.download_button(
                    label="Download MP3",
                    data=audio_data,
                    file_name=filename,
                    mime="audio/mp3"
                )

    st.write("---")
    st.write("""
    **Notes**:
    - For "Audiobook," the app uses the original text from the PDF, no summarization.
    - For "Podcast," it chunk-summarizes each chapter, then does an optional 
      second-pass summary (chain-of-summaries).
    - If using Ollama, confirm that it's serving at `ollama serve --port 11434` and 
      that you have the correct model pulled (e.g. `ollama pull llama2`).
    """)

if __name__ == "__main__":
    main()
