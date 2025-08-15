import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

os.environ["GROQ_API_KEY"] = "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"

def get_context_type(query):
    programming_languages = {"python", "java", "c++", "javascript", "go", "rust"}
    return "programming language" if query.lower() in programming_languages else "character"

def generate_response(topic, context, tone=None):
    if context == "character":
        template = """
        Write a {tone} story outline about {topic}.
        Structure the response into the following points:
        - **Introduction:** (Brief introduction to the character)
        - **Background:** (Character's history or origin)
        - **Conflict:** (What challenge or adventure they face)
        - **Climax:** (How the story reaches its peak)
        - **Conclusion:** (How the story ends)
        Keep the response concise.
        """
    elif context == "programming language":
        template = """
        Provide a structured explanation of {topic} covering:
        - **Introduction:** (What is {topic}?)
        - **Key Features:** (Main characteristics of {topic})
        - **Use Cases:** (Where and why it is used)
        - **Example Code:** (Basic code snippet)
        Keep the explanation clear and concise.
        """

    prompt = PromptTemplate(
        input_variables=["topic", "tone"] if context == "character" else ["topic"],
        template=template
    )

    llm = ChatGroq(model_name="llama3-8b-8192")
    response = llm.invoke(prompt.format(topic=topic, tone=tone) if context == "character" else prompt.format(topic=topic))

    response_text = response.content if hasattr(response, "content") else str(response)
    return response_text

st.set_page_config(page_title="Content Generator", layout="wide")
st.title("📝 AI-Powered Content Generator")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Select a Page", ["Home", "Generate Content"])

if page == "Home":
    st.write("### Welcome to the AI-Powered Content Generator!")
    st.write("Select 'Generate Content' from the sidebar to get started.")

elif page == "Generate Content":
    st.header("Generate AI-Powered Content")
    topic = st.text_input("Enter a topic (e.g., name, programming language):")
    if topic:
        context = get_context_type(topic)
        tone = st.selectbox("Select a tone:", ["Inspirational", "Humorous", "Suspenseful"]) if context == "character" else None
        
        if st.button("Generate Response"):
            with st.spinner("Generating..."):
                response = generate_response(topic, context, tone)
                st.text_area("Generated Response:", response, height=300)
                
st.header("Structured Text Generation")
model = ChatGroq(model_name="llama3-8b-8192")
query = st.text_input("Enter a topic for structured response:")
if st.button("Generate Structured Response"):
    with st.spinner("Generating..."):
        response = model.invoke(query)
        structured_response = f"""
        **Generated Response for '{query}':**
        
        {response.content if hasattr(response, "content") else response}
        """
        st.text_area("Response:", structured_response, height=300)
