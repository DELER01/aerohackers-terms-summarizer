from flask import Flask, request, jsonify, render_template
import os
import asyncio
import aiohttp
import pdfplumber
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, template_folder="templates")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# API configuration
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

# NDPR keywords
ndpr_keywords = {
    "consent": ["consent", "rights"],
    "security": ["security", "delete"],
    "collection": ["collect", "third party"],
    "regulation": ["ndpr", "nitda", "legal", "law"]
}

# In-memory cache
cache = {}

# Clean text function
def clean_text(text):
    soup = BeautifulSoup(text, 'html.parser') if '<' in text else None
    if soup:
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
    return re.sub(r'\[\w\]|call\(this\);', '', text).strip()

# Split text into chunks
def split_into_chunks(text, max_chars=2000):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks[:3] if len(chunks) > 2 else chunks

# Filter NDPR sentences
def filter_ndpr_sentences(text, keywords):
    sentences = text.split('.')
    categorized = defaultdict(list)
    for s in sentences:
        s = clean_text(s).strip()
        if s:
            for category, kws in keywords.items():
                if any(k.lower() in s.lower() for k in kws):
                    categorized[category].append(s)
                    break
    return categorized

# Summarize text asynchronously
async def summarize_text(session, text, max_length=15):
    cache_key = hash(text)
    if cache_key in cache:
        return cache[cache_key]
    for attempt in range(3):
        try:
            async with session.post(API_URL, headers=headers, json={"inputs": text, "parameters": {"max_length": max_length}}, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    summary = (await response.json())[0]["summary_text"]
                    cache[cache_key] = summary
                    return clean_text(summary).split('.')[0] + '.'
        except Exception as e:
            if attempt == 2:
                return f"API error after retries: {str(e)}"
            await asyncio.sleep(2 ** attempt)
    return "API error: Max retries exceeded"

# Fetch content from URL
def fetch_url_content(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return clean_text(response.text)
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

# Summarize route
@app.route('/summarize', methods=['POST'])
def summarize():
    ndpr_mode = request.form.get('ndpr') == 'on'
    input_type = request.form.get('input_type')

    if input_type == 'pdf':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.pdf'):
            return jsonify({"error": "Only PDF files are allowed"}), 400
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text += page_text + "\n"
            if not text:
                return jsonify({"error": "No text extracted from PDF"}), 400
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    elif input_type == 'url':
        url = request.form.get('url')
        if not url:
            return jsonify({"error": "No URL provided"}), 400
        text = fetch_url_content(url)
        if text.startswith("Error"):
            return jsonify({"error": text}), 400
    elif input_type == 'text':
        text = request.form.get('pasted_text', '').strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
    else:
        return jsonify({"error": "Invalid input type"}), 400

    text = clean_text(text)
    if ndpr_mode:
        categorized_text = filter_ndpr_sentences(text, ndpr_keywords)
        return jsonify({
            "categories": {k: [clean_text(v[0])] for k, v in categorized_text.items() if v},
            "note": ""
        })
    else:
        chunks = split_into_chunks(text)
        async def process_chunks():
            async with aiohttp.ClientSession() as session:
                tasks = [summarize_text(session, chunk, max_length=30) for chunk in chunks]
                summaries = await asyncio.gather(*tasks)
                return [s for s in summaries if not s.startswith("API error")]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        summaries = loop.run_until_complete(process_chunks())
        return jsonify({
            "summary": " ".join(summaries[:2]) or "No summary generated",
            "note": "Limited to 1 chunk; compare to NDPR manually"
        })

# Home route
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)