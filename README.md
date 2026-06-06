# Creative Content Generator using AI Libraries

A full-stack AI-powered Creative Content Generator application built using Python, Streamlit, LLaMA 3, Whisper, Pillow, and ChromaDB. The application generates structured content such as stories, captions, and summaries from text, images, and videos.

---

## Features

* AI-Based Content Generation
* Story, Caption, and Summary Generation
* Video Transcription using Whisper
* Video Download Support using yt-dlp
* Image Processing using Pillow
* Text Generation using LLaMA 3
* ChromaDB Storage and Retrieval
* Interactive Streamlit User Interface
* Multimedia Input Support (Text, Images, Videos)

---

## Tech Stack

### Frontend

* Streamlit
* HTML
* CSS

### Backend

* Python

### AI & Libraries

* LLaMA 3
* Whisper
* ChromaDB
* Pillow
* yt-dlp

---

## Project Structure

```text id="b3j4l5"
Content-Generator/
|
├── app.py
├── requirements.txt
├── static/
├── templates/
├── uploads/
├── generated_content/
└── README.md
```

---

## Application Workflow

1. User uploads text, image, or video input.
2. Videos are processed and transcribed using Whisper and yt-dlp.
3. Images are processed using Pillow.
4. LLaMA 3 analyzes the input and generates content.
5. Generated content is stored and managed using ChromaDB.
6. Streamlit displays the generated output through an interactive interface.

---

## Installation

### Clone Repository

```bash id="k7m8n9"
git clone https://github.com/abhinayapravallika/Content-Generator.git

cd Content-Generator
```

### Create Virtual Environment

```bash id="r5t6y7"
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash id="u8i9o0"
venv\Scripts\activate
```

### Install Dependencies

```bash id="p1q2w3"
pip install -r requirements.txt
```

### Run Application

```bash id="a4s5d6"
streamlit run app.py
```

### Open Browser

```text id="f7g8h9"
http://localhost:8501
```

---

## Future Enhancements

* User Authentication
* Voice Input Support
* Multiple Language Support
* Cloud Deployment
* AI Chatbot Integration
* Export Generated Content as PDF
* Advanced Content Customization

---

## Author

RAMENA ABHINAYA PRAVALLIKA

B.Tech – Computer Science and Engineering

Full Stack Developer | AI Enthusiast

GitHub: https://github.com/abhinayapravallika

LinkedIn: https://www.linkedin.com/in/abhinaya-pravallika-ramena-235130261/
