import streamlit as st
from pytubefix import YouTube
import os
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModelForCausalLM, AutoTokenizer
import re
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from model_loader import load_blip_model
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# ----------------- Transcript Generation Class -----------------
class TranscriptGen:
    def __init__(self, path, model_p="models/whisper_model_p", model_m="models/whisper_model_m",
                 sampling_rate=16000, chunk_size=30):
        self.path = path
        self.sampling_rate = sampling_rate
        self.chunk_size = chunk_size
        self.processor = WhisperProcessor.from_pretrained(model_p)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_m)

    def generate_transcript(self):
        # Check if file exists before processing
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Error: File '{self.path}' not found. Please check the filename and path.")

        # Load audio file
        print(f"Loading audio file: {self.path}")
        speech_array, sr = librosa.load(self.path, sr=self.sampling_rate)

        chunk_samples = self.chunk_size * self.sampling_rate
        total_samples = len(speech_array)
        transcript = ""

        for i in range(0, total_samples, chunk_samples):
            chunk = speech_array[i: i + chunk_samples]
            input_features = self.processor(chunk, sampling_rate=self.sampling_rate, return_tensors="pt").input_features

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features)

            chunk_transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            transcript += chunk_transcription + " "

        return transcript.strip()
# ------------------------------------------------------------------------------------------------------------
def generate_multiple_captions_with_llama(image_path, processor, model, llm, num_captions=5):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        output = model.generate(
            **inputs,
            num_beams=num_captions,
            num_return_sequences=num_captions,
            max_length=50,
            early_stopping=True
        )
        blip_captions = [processor.decode(out, skip_special_tokens=True) for out in output]
        
        prompt = f"""Generate {num_captions} short, catchy, and creative captions (each under 8 words) for an image described as: '{blip_captions[0]}'. 
        Keep the captions crisp, engaging, and suitable for social media or titles. Avoid long sentences."""
        
        llm_response = llm.invoke(prompt)
        c = llm_response.content
        pattern = r'\d+\.\s*"(.*?)"'
        captions = re.findall(pattern, c)
        return captions

    except Exception as e:
        return [f"Error generating captions: {str(e)}"]
# ----------------- ----------------------------------------------------
def get_context_type(query):
    programming_languages = {
    # Programming Languages
    "python", "java", "c", "c++", "c#", "javascript", "typescript", "go", "rust", "ruby", "perl",
    "swift", "kotlin", "r", "dart", "php", "scala", "haskell", "lua", "matlab", "fortran",
    "shell", "bash", "objective-c", "groovy", "lisp", "elixir", "clojure", "f#", "ada", "julia",
    "cobol", "pascal", "abap", "prolog", "smalltalk", "scheme", "apl", "erlang", "modula-2",
    "ocaml", "rexx", "tcl", "crystal", "forth", "hack", "idl", "icon", "j", "ml", "x10", "xquery",
    "yaml"

    # Web Technologies
    "html", "css", "react", "angular", "vue", "node.js", "express.js", "django", "flask",
    "spring boot", "next.js", "svelte", "tailwind", "bootstrap", "meteor.js", "nuxt.js",
    "gatsby.js", "hugo", "jekyll", "blazor", "webassembly", "stencil.js", "alpine.js"

    # Databases
    "mysql", "postgresql", "mongodb", "sqlite", "oracle", "redis", "cassandra", "neo4j",
    "sql server", "microsoft sql server", "firebase", "dynamodb", "cockroachdb", "mariadb",
    "tarantool", "amazon aurora", "apache derby", "influxdb", "timescaledb", "clickhouse",
    "tidb", "voltdb",

    # Cloud Platforms
    "aws", "azure", "gcp", "ibm cloud", "oracle cloud", "digitalocean", "linode", "heroku",
    "netlify", "vercel", "cloudflare",

    # DevOps & CI/CD Tools
    "docker", "kubernetes", "jenkins", "terraform", "ansible", "prometheus", "grafana",
    "git", "github", "gitlab", "helm", "openshift", "nomad", "puppet", "chef", "saltstack",
    "argocd", "fluxcd", "spinnaker", "consul",

    # Operating Systems
    "windows", "linux", "macos", "android", "ios", "ubuntu", "redhat", "debian", "fedora",
    "centos", "opensuse", "arch linux", "kali linux", "solaris", "aix", "freebsd", "haiku",

    # Networking Concepts
    "tcp/ip", "http", "https", "dns", "vpn", "firewall", "load balancer", "subnet", "proxy",
    "dhcp", "bgp", "nat", "qos", "snmp", "ipsec", "mpls", "socks proxy",

    # Cybersecurity Topics
    "encryption", "firewalls", "penetration testing", "zero trust", "malware", "phishing",
    "ransomware", "ethical hacking", "soc", "ids/ips", "siem", "cyber threat intelligence",
    "dark web monitoring",

    # AI/ML Technologies
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow", "pytorch",
    "scikit-learn", "keras", "fast.ai", "opencv", "deepmind", "lightgbm", "hugging face transformers",
    "spacy", "gensim", "xgboost", "pandas", "numpy", "matplotlib", "seaborn", "power bi", "tableau",
    "dask", "polars", "vaex", "altair", "plotly", "bokeh", "databricks",

    # Software Development Methodologies
    "agile", "scrum", "devops", "waterfall", "kanban", "extreme programming", "lean development",
    "rad", "feature-driven development",

    # Blockchain Technologies
    "bitcoin", "ethereum", "smart contracts", "web3", "blockchain", "solana", "polkadot",
    "hyperledger", "binance smart chain", "cosmos", "avalanche", "tezos"

    #full_stack_development
    "full stack", "mern stack", "mean stack", "lamp stack", "django full stack", "ruby on rails full stack"
    
    #cse_subjects 
    "data structures", "algorithms", "computer networks", "operating systems", "database management systems",
    "software engineering", "artificial intelligence", "computer organization", "compiler design",
    "web development", "mobile app development", "cryptography", "big data", "cloud computing"
    
    
}
    return "programming language" if query.lower() in programming_languages else "character"

def generate_response(topic, context, tone=None):
    if context == "character":
        template = """
        Write a {tone} story about {topic}.
        The story should be engaging, creative, and follow a structured flow with an introduction,
        middle, and end. Keep it under 450 words.
        """
    elif context == "programming language":
        template = """
        Provide an explanation of {topic}, covering its basics, use cases, and why it is important.
        Include key features and examples. Keep the explanation concise and informative.
        """

    prompt = PromptTemplate(
        input_variables=["topic", "tone"] if context == "character" else ["topic"],
        template=template
    )
    
    llm = ChatGroq(model_name="llama3-8b-8192")
    response = llm.invoke(prompt.format(topic=topic, tone=tone) if context == "character" else prompt.format(topic=topic))

    response_text = response.content if hasattr(response, "content") else str(response)
    return response_text
# ----------------- Streamlit Navigation -----------------
st.set_page_config(page_title="Multi-Task AI App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose a Task:", ["🎙️ YouTube Audio Transcriber", "📝 Text Generation", "🖼️ Image Caption Generation","🔊 Story Generation"])

if page == "🎙️ YouTube Audio Transcriber":
    st.title("🎙️ YouTube Audio Downloader & Transcriber")
    tab1, tab2 = st.tabs(["📥 Download Audio", "📝 Transcribe Audio"])

    with tab1:
        st.header("Download YouTube Audio as .mp3")
        url = st.text_input("Enter the URL of the YouTube video:")
        destination = st.text_input("Enter destination folder (default: current directory):", ".")

        if st.button("Download Audio"):
            if url:
                try:
                    yt = YouTube(url)
                    video = yt.streams.filter(only_audio=True).first()
                    st.info("Downloading...")
                    out_file = video.download(output_path=destination)
                    base, ext = os.path.splitext(out_file)
                    new_file = base + '.mp3'
                    os.rename(out_file, new_file)
                    st.success(f"{yt.title} has been successfully downloaded as {new_file}.")
                    st.session_state["downloaded_file"] = new_file
                except Exception as e:
                    st.error(f"❌ Error: {e}")
            else:
                st.warning("⚠️ Please provide a valid URL.")

    with tab2:
        st.header("Transcribe Downloaded Audio")
        audio_file = st.session_state.get("downloaded_file", None)

        if audio_file:
            st.audio(audio_file, format="audio/mp3")
            if st.button("Generate Transcript"):
                with st.spinner("Transcribing audio... Please wait."):
                    try:
                        transcriber = TranscriptGen(audio_file)
                        transcript = transcriber.generate_transcript()
                        st.success("✅ Transcription complete!")
                        st.text_area("Generated Transcript:", transcript, height=300)
                    except Exception as e:
                        st.error(f"❌ Error during transcription: {e}")
        else:
            st.info("ℹ️ Please download an audio file from Tab 1 before transcription.")

elif page == "📝 Text Generation":
# Initialize the model
    os.environ["GROQ_API_KEY"] = "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"
    model = ChatGroq(model="llama3-8b-8192")
    user_input = st.text_input("Enter a topic (e.g., moon):")
    if st.button("Generate Text"):
        with st.spinner("Generating response..."):
            response = model.invoke(user_input)
            st.success("Generated Text:")
            st.write(response.content)

processor, model = load_blip_model()


if page == "🖼️ Image Caption Generation":
    st.title("🖼️ Image Caption Generator")
    uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        llm = ChatGroq(model_name="llama3-8b-8192")
        captions = generate_multiple_captions_with_llama(
            uploaded_image, processor, model, llm, num_captions=5
        )
        
        if captions:
            st.subheader("Generated Captions:")
            for i, caption in enumerate(captions, start=1):
                st.write(f"{i}. {caption}")
        else:
            st.warning("No captions generated. Please try again.")
elif page == "🔊 Story Generation":
    st.header("Story Generation")
    topic = st.text_input("Enter a topic (e.g., Name, Programming Language):")
    
    if topic:
        context = get_context_type(topic)
        tone = st.text_input("Enter a tone (e.g., Inspirational, Humorous, Suspenseful, etc.):") if context == "character" else None
        
        if st.button("Generate Response"):
            with st.spinner("Generating..."):
                response = generate_response(topic, context, tone)
                st.text_area("Generated Response:", response, height=300)



