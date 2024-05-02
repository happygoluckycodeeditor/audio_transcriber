import streamlit as st
import whisper

# Initialize the Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio_file):
    # Load the audio file
    audio = whisper.load_audio(audio_file)
    audio = whisper.pad_or_trim(audio)

    # Make prediction
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions()
    result = model.decode(mel, options)

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

