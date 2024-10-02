# **PDF Key Concept Summarization Project using BART and Hugging Face APIs**
# Run this notebook on Google Colab

# --- SECTION 1: Install Dependencies ---
!pip
install
transformers
gradio
pymupdf  # PyMuPDF for reading PDFs

# --- SECTION 2: Import Libraries ---
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import fitz  # PyMuPDF library to extract text from PDF
import gradio as gr


# --- SECTION 3: Function to Extract Text from PDF ---

# Function to extract text from uploaded PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


# --- SECTION 4: Load BART Model and Tokenizer ---

# Use the pre-trained BART model from Hugging Face
model_name = 'facebook/bart-large-cnn'  # BART model for summarization
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# --- SECTION 5: Define Summarization Pipeline ---

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


# Function to summarize extracted text from the PDF
def summarize_text(text):
    if len(text) > 1024:  # Handle large text inputs by truncating for summarization
        text = text[:1024]

    summary = summarizer(text, max_length=150, min_length=40, do_sample=False)
    return summary[0]['summary_text']


# --- SECTION 6: Define Gradio App ---

# Function to handle file upload, extract text, and generate key concepts summary
def summarize_pdf(pdf_file):
    # Step 1: Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(pdf_file.name)

    # Step 2: Summarize extracted text
    summary = summarize_text(pdf_text)
    return summary


# Gradio interface for uploading PDF and summarizing
interface = gr.Interface(
    fn=summarize_pdf,
    inputs=gr.File(label="Upload PDF"),  # Use gr.File for PDF upload
    outputs=gr.Textbox(label="Key Concept Summary"),  # Use gr.Textbox for output
    title="Key Concept Extractor (BART-based)"
)

# Launch the Gradio app
interface.launch(share=True)
