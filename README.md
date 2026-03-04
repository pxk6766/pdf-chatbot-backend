FRONTEND
•	Framework: Vanilla JavaScript 
•	Markup: HTML5
•	Styling: Custom CSS 
•	HTTP Client: Fetch API
•	Rendering: Client-side DOM manipulation
•	Deployment: Vercel (serverless hosting)
•	CI/CD: GitHub → Vercel auto-deployment
________________________________________
BACKEND
•	Framework: FastAPI (Python web framework)
•	Server: Uvicorn (ASGI server)
•	API Architecture: RESTful API
•	Endpoints: 
o	GET / - Health check
o	POST /upload - PDF file upload & processing
o	POST /ask - Question-answering with AI
•	Middleware: CORS (Cross-Origin Resource Sharing)
•	Deployment: Hugging Face Spaces (Docker container)
•	Containerization: Docker + Docker Compose
________________________________________
VECTOR DATABASE
•	Database: Pinecone 
o	Type: Cloud-hosted vector database
o	Dimension: 384 (matching embedding model)
•	Storage Strategy: "delete_all" approach 
o	Clears previous PDF on each upload
o	Supports unlimited users (one PDF at a time)
o	Free tier: 100,000 vectors
________________________________________
PDF PROCESSING
•	Library: PyPDF2 
o	Purpose: Extract text from PDF files
o	Processing: Page-by-page text extraction
•	Text Chunking: Custom Python function 
o	Chunk size: 1000 characters
o	Overlap: 200 characters
o	Purpose: Break large documents into searchable segments
________________________________________
DATA FLOW ARCHITECTURE
User Upload PDF
    ↓
Frontend (Vercel) → Backend (Hugging Face)
    ↓
PyPDF2 extracts text
    ↓
Custom chunking (1000 chars, 200 overlap)
    ↓
Sentence Transformers (local embeddings)
    ↓
Pinecone (cloud storage) ← delete_all=True
    ↓
Ready for queries

User Asks Question
    ↓
Frontend → Backend
    ↓
Sentence Transformers (query embedding)
    ↓
Pinecone (similarity search, top 3 chunks)
    ↓
Gemini 2.5 Flash (generate answer)
    ↓
Backend → Frontend → User sees answer
________________________________________
ENVIRONMENT & CONFIGURATION
•	Environment Variables: 
o	GEMINI_API_KEY - Google AI API key
o	PINECONE_API_KEY - Vector database API key
•	Config Management: python-dotenv
•	Environment Isolation: Python virtual environment (venv)
________________________________________
