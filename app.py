import streamlit as st
import whisper
import tempfile
import os

# Initialize the Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio_file):
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + audio_file.name.split('.')[-1]) as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name

    # Decode the audio file with timing information
    result = model.transcribe(tmp_path)

    # Clean up the temporary file
    os.remove(tmp_path)

    # Format transcription to SRT-like format
    srt_output = format_to_srt(result['segments'])
    return srt_output

def format_to_srt(segments):
    srt_list = []
    for i, segment in enumerate(segments):
        start_time = format_time(segment['start'])
        end_time = format_time(segment['end'])
        text = segment['text']
        srt_list.append(f"{i+1}\n{start_time} --> {end_time}\n{text}\n")
    return "\n".join(srt_list)

def format_time(seconds):
    #Converts the text into SRT time format. I do not know how this part works lol
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:06.3f}".replace('.', ',')

# Streamlit interface
st.title('Audio Transcription using Whisper with SRT Format')
audio_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3', 'ogg', 'm4a'])

if audio_file is not None:
    # Display audio player
    st.audio(audio_file, format='audio/wav')

    # Transcribe button
    if st.button('Transcribe Audio'):
        with st.spinner('Transcribing...'):
            transcription = transcribe_audio(audio_file)
            st.write("Transcription (SRT Format):")
            st.text_area("", transcription, height=150)
