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

    # Load the audio file
    audio = whisper.load_audio(tmp_path)
    audio = whisper.pad_or_trim(audio)

    # Make prediction
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = model.decode(mel, options)

    # Clean up the temporary file
    os.remove(tmp_path)

    return result.text

# Streamlit interface
st.title('Audio Transcription using Whisper')
audio_file = st.file_uploader("Choose an audio file...", type=['wav', 'mp3', 'ogg', 'm4a'])

if audio_file is not None:
    # Display audio player
    st.audio(audio_file, format='audio/wav')

    # Transcribe button
    if st.button('Transcribe Audio'):
        with st.spinner('Transcribing...'):
            transcription = transcribe_audio(audio_file)
            st.write("Transcription:")
            st.text_area("", transcription, height=150)
