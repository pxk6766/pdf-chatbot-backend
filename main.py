from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import time

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash') 

# Pinecone setup
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "pdf-chatbot"

print("🔍 Checking Pinecone index...")
existing = pc.list_indexes().names()
if INDEX_NAME not in existing:
    print(f"📝 Creating index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(INDEX_NAME).status['ready']:
        time.sleep(1)
    print("✅ Index ready!")
else:
    print("✅ Index exists!")

index = pc.Index(INDEX_NAME)

print("📦 Loading sentence transformer...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!")

# Simple chunking
def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]

def safe_index_operation(func, *args, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                wait = (attempt + 1) * 3
                print(f"⚠️ Pinecone warming up... Retrying in {wait}s")
                time.sleep(wait)
            else:
                raise e

class QuestionRequest(BaseModel):
    question: str

@app.get("/")
def root():
    return {"message": "PDF Chatbot API running! 🚀"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(400, "Only PDF files!")
        
        print(f"📄 Processing: {file.filename}")
        
        pdf_reader = PdfReader(file.file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text() + "\n"
        
        if not pdf_text.strip():
            raise HTTPException(400, "No text found in PDF")
        
        chunks = chunk_text(pdf_text)
        embeddings = embedding_model.encode(chunks).tolist()
        
        # Clears old data safely
        try:
            stats = safe_index_operation(index.describe_index_stats)
            if stats.get('total_vector_count', 0) > 0:
                safe_index_operation(index.delete, delete_all=True)
        except:
            pass

        vectors = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            safe_chunk = str(chunk)[:30000] 
            vectors.append({
                "id": f"chunk_{i}", 
                "values": emb, 
                "metadata": {"text": safe_chunk, "source": str(file.filename)}
            })

        for i in range(0, len(vectors), 100):
            safe_index_operation(index.upsert, vectors=vectors[i:i+100])
        
        print("✅ Done!")
        return {"message": f"✅ {file.filename} processed!", "chunks": len(chunks)}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(500, str(e))

@app.post("/ask")
async def ask(request: QuestionRequest):
    try:
        question = request.question.strip()
        if not question:
            raise HTTPException(400, "Empty question")
        
        stats = safe_index_operation(index.describe_index_stats)
        if stats.get('total_vector_count', 0) == 0:
            return {"answer": "⚠️ Upload a PDF first!"}

        q_emb = embedding_model.encode([question]).tolist()[0]
        
        results = safe_index_operation(index.query, vector=q_emb, top_k=3, include_metadata=True)
        
        if not results['matches']:
            return {"answer": "No relevant info found"}

        context = "\n\n".join([m['metadata']['text'] for m in results['matches']])
        
        prompt = f"Based on this PDF content, answer the question.\nPDF:{context}\nQuestion: {question}\nAnswer:"
        
        response = model.generate_content(prompt)
        return {"answer": response.text}
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return {"answer": f"Error: {e}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
