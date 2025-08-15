# import streamlit as st
# from pytubefix import YouTube
# import os
# import torch
# import librosa
# from transformers import WhisperProcessor, WhisperForConditionalGeneration

# # ----------------- Transcript Generation Class -----------------
# class TranscriptGen:
#     def __init__(self, path, model_p="models/whisper_model_p", model_m="models/whisper_model_m",
#                  sampling_rate=16000, chunk_size=30):
#         self.path = path
#         self.sampling_rate = sampling_rate
#         self.chunk_size = chunk_size  # In seconds
#         self.processor = WhisperProcessor.from_pretrained(model_p)
#         self.model = WhisperForConditionalGeneration.from_pretrained(model_m)

#     def generate_transcript(self):
#         speech_array, sr = librosa.load(self.path, sr=self.sampling_rate)
#         chunk_samples = self.chunk_size * self.sampling_rate
#         total_samples = len(speech_array)
#         transcript = ""

#         for i in range(0, total_samples, chunk_samples):
#             chunk = speech_array[i: i + chunk_samples]
#             input_features = self.processor(chunk, sampling_rate=self.sampling_rate, return_tensors="pt").input_features

#             with torch.no_grad():
#                 predicted_ids = self.model.generate(input_features)

#             chunk_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
#             transcript += chunk_transcription + " "

#         return transcript.strip()

# # ----------------- Streamlit Interface -----------------
# st.set_page_config(page_title="YouTube Audio Transcriber", layout="wide")
# st.title("🎙️ YouTube Audio Downloader & Transcriber")

# # Tabs for Download and Transcription
# tab1, tab2 = st.tabs(["📥 Download Audio", "📝 Transcribe Audio"])

# with tab1:
#     st.header("Download YouTube Audio as .mp3")
#     url = st.text_input("Enter the URL of the YouTube video:")
#     destination = st.text_input("Enter destination folder (default: current directory):", ".")

#     if st.button("Download Audio"):
#         if url:
#             try:
#                 yt = YouTube(url)
#                 video = yt.streams.filter(only_audio=True).first()
#                 st.info("Downloading...")
#                 out_file = video.download(output_path=destination)
#                 base, ext = os.path.splitext(out_file)
#                 new_file = base + '.mp3'
#                 os.rename(out_file, new_file)
#                 st.success(f"{yt.title} has been successfully downloaded as {new_file}.")
#                 st.session_state["downloaded_file"] = new_file
#             except Exception as e:
#                 st.error(f"❌ Error: {e}")
#         else:
#             st.warning("⚠️ Please provide a valid URL.")

# with tab2:
#     st.header("Transcribe Downloaded Audio")
#     audio_file = st.session_state.get("downloaded_file", None)

#     if audio_file:
#         st.audio(audio_file, format="audio/mp3")
#         if st.button("Generate Transcript"):
#             with st.spinner("Transcribing audio... Please wait."):
#                 try:
#                     transcriber = TranscriptGen(audio_file)
#                     transcript = transcriber.generate_transcript()
#                     st.success("✅ Transcription complete!")
#                     st.text_area("Generated Transcript:", transcript, height=300)
#                 except Exception as e:
#                     st.error(f"❌ Error during transcription: {e}")
#     else:
#         st.info("ℹ️ Please download an audio file from Tab 1 before transcription.")


import streamlit as st
from pytubefix import YouTube
import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------------- Transcript Generation Class -----------------
class TranscriptGen:
    def __init__(self, path, model_p="models/whisper_model_p", model_m="models/whisper_model_m",
                 sampling_rate=16000, chunk_size=30):
        self.path = path
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size  # In seconds
        self.processor = WhisperProcessor.from_pretrained(model_p)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_m)

    def generate_transcript(self):
        try:
            speech_array, sr = librosa.load(self.path, sr=self.sampling_rate)
            chunk_samples = self.chunk_size * self.sampling_rate
            total_samples = len(speech_array)
            transcript = ""

            for i in range(0, total_samples, chunk_samples):
                chunk = speech_array[i: i + chunk_samples]
                input_features = self.processor(chunk, sampling_rate=self.sampling_rate, return_tensors="pt").input_features

                with torch.no_grad():  # Reduce memory usage
                    predicted_ids = self.model.generate(input_features)

                chunk_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                transcript += chunk_transcription + " "

            return transcript.strip()

        except Exception as e:
            return f"Error during transcription: {e}"

# ----------------- Streamlit Interface -----------------
st.set_page_config(page_title="YouTube Audio Transcriber", layout="wide")
st.title("🎙️ YouTube Audio Downloader & Transcriber")

# Tabs for Download and Transcription
tab1, tab2 = st.tabs(["📥 Download Audio", "📝 Transcribe Audio"])

with tab1:
    st.header("Download YouTube Audio as .mp3")
    url = st.text_input("Enter the URL of the YouTube video:")
    destination = st.text_input("Enter destination folder (default: current directory):", ".")

    if st.button("Download Audio"):
        if url.strip():
            try:
                yt = YouTube(url)
                video = yt.streams.filter(only_audio=True).first()
                st.info("Downloading... Please wait.")
                out_file = video.download(output_path=destination)
                base, ext = os.path.splitext(out_file)
                new_file = base + ".mp3"
                os.rename(out_file, new_file)

                st.success(f"✅ {yt.title} has been successfully downloaded as {new_file}.")
                st.session_state["downloaded_file"] = new_file  # Store file path in session state

            except Exception as e:
                st.error(f"❌ Download Error: {e}")
        else:
            st.warning("⚠️ Please enter a valid YouTube URL.")

with tab2:
    st.header("Transcribe Downloaded Audio")
    audio_file = st.session_state.get("downloaded_file", None)

    if audio_file and os.path.exists(audio_file):
        st.audio(audio_file, format="audio/mp3")
        if st.button("Generate Transcript"):
            with st.spinner("📝 Transcribing audio... Please wait."):
                transcriber = TranscriptGen(audio_file)
                transcript = transcriber.generate_transcript()

                if "Error" not in transcript:
                    st.success("✅ Transcription complete!")
                    st.text_area("Generated Transcript:", transcript, height=300)
                else:
                    st.error(transcript)  # Display error message from transcription
    else:
        st.info("ℹ️ No audio file found. Please download an audio file in Tab 1 before transcription.")
