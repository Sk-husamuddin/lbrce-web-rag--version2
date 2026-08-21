# LBRCE Reference Desk — Version 2

A grounded question-answering assistant for **Lakireddy Bali Reddy College of Engineering (LBRCE)**. This repository contains an independent version-2 implementation using FastAPI, LangGraph, local BGE embeddings, Pinecone, Groq-compatible answer generation, controlled Tavily retrieval, and a Next.js frontend.

The project is designed for local development, Docker Compose execution, academic demonstration, and continued RAG experimentation. It uses a dedicated local-BGE vector configuration and does not depend on any other deployment.

> The assistant should prefer an honest no-match response over inventing information, mixing unrelated documents, or presenting historical information as current.

## Project Overview

LBRCE Reference Desk is a retrieval-augmented generation application. A user submits a natural-language question through the frontend. The backend identifies the question’s intent, generates a retrieval-focused query using the local BGE model, searches approved LBRCE evidence in Pinecone, evaluates the evidence, and produces a grounded answer.

The system can answer questions about LBRCE webpages, departments, admissions, facilities, placements, student lists, timetables, transportation, examination resources, R23 regulations, and R23 syllabus documents. Official timetable images, timetable PDFs, regulation PDFs, and syllabus PDFs can be returned as direct visual resources.

## Architecture

```text
                         User browser
                              |
                              v
                   Next.js / React frontend
                              |
                    POST /chat and GET /health
                              |
                              v
                   FastAPI application backend
                              |
                              v
                     LangGraph workflow
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
  Query planner        Local BGE query       Intent metadata
  and policies         embedding model      filters and guards
                              |
                              v
                    Pinecone vector search
                 lbrce-local-bge-index
                 lbrce_local_bge_full_v1
                              |
                              v
                    Evidence evaluation
                              |
               +--------------+--------------+
               |                             |
               v                             v
       Grounded Pinecone path        Controlled fallback path
                                      Tavily or approved contact pages
               |                             |
               +--------------+--------------+
                              v
                    Context assembly
                              |
                              v
                   Groq answer generation
                              |
                              v
                 Answer, sources, and visual resources
```

### Request flow

The frontend sends a question to `POST /chat`. The planner in `backend/graph/nodes.py` identifies the user’s intent and extracts constraints such as department, semester, section, route code, academic year, document type, or requested PDF.

The retrieval node creates a local BGE query embedding and searches the configured Pinecone namespace. For protected topics, metadata filters are applied before evidence is accepted. The evidence evaluator prevents a weak or unrelated semantic match from being treated as sufficient for a general question.

The context assembler removes unrelated evidence and prepares the final context. The answer generator uses the configured Groq-compatible model after evidence selection. The groundedness check prevents unsupported answers and controls whether an approved fallback is allowed.

## Technology Stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS | User interface and resource display |
| API | FastAPI, Uvicorn, Pydantic Settings | HTTP service and configuration |
| Orchestration | LangGraph, LangChain Core | Planner, retrieval, evaluation, and generation workflow |
| Embeddings | `BAAI/bge-large-en-v1.5` through Sentence Transformers | Local 1024-dimensional query and document embeddings |
| Vector search | Pinecone | Similarity search and metadata-filtered retrieval |
| Answer generation | Groq OpenAI-compatible API | Grounded natural-language response generation |
| Controlled web fallback | Tavily | Restricted fallback for approved cases |
| Packaging | Docker and Docker Compose | Reproducible local backend environment |
| Official source | LBRCE website | Approved webpages, PDFs, images, and tables |

## Pinecone and Embedding Configuration

The version-2 system uses a dedicated migration index and namespace:

```text
PINECONE_INDEX_NAME=lbrce-local-bge-index
PINECONE_NAMESPACE=lbrce_local_bge_full_v1
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024
```

The local BGE model is used for both document ingestion and user-query embeddings. The query path applies the BGE search instruction prefix, while passage embeddings use the document text without the query prefix. This asymmetric behavior is required for correct BGE retrieval.

Pinecone hosted embedding inference is not used for the local-BGE migration path. User questions are embedded temporarily for retrieval and are not stored as new document vectors.

## Repository Structure

```text
.
├── backend/
│   ├── main.py                         # FastAPI application and health route
│   ├── api/routes/chat.py              # Chat request and response endpoint
│   ├── config/settings.py              # Environment-backed settings
│   ├── graph/
│   │   ├── graph.py                    # LangGraph topology
│   │   ├── state.py                    # GraphState fields
│   │   ├── nodes.py                    # Planner, retrieval, policies, and generation
│   │   └── constants.py                # Shared answer constants
│   ├── embedding/
│   │   ├── __init__.py                 # Configured embedding factory
│   │   └── local_bge_embedding_generator.py
│   ├── retrieval/
│   │   └── rag.py                      # Retrieval adapter and legacy RAG path
│   ├── indexing/
│   │   └── pinecone_indexer.py         # Pinecone connection and upsert helpers
│   └── ingestion/
│       ├── html_parser.py              # HTML extraction
│       ├── chunker.py                  # Chunk creation and metadata preservation
│       └── selected_resources.py       # Approved resource handling
├── frontend/
│   ├── app/                            # Next.js routes and layout
│   ├── components/                     # Chat, sources, and visual resources
│   ├── lib/api.ts                      # Browser API client
│   └── package.json                    # Frontend dependencies and scripts
├── scripts/
│   ├── run_ingestion.py                # General and selective ingestion entry point
│   ├── migrate_html_registry_local_bge.py
│   ├── migrate_academic_resources_local_bge.py
│   ├── ingest_student_lists.py
│   ├── ingest_transportation_routes_local_bge.py
│   ├── backfill_regulation_metadata.py
│   ├── r23_regulations_manifest.json
│   ├── academic_resources_manifest.json
│   └── timetable_ingestion_registry.json
├── tests/                              # Routing, retrieval, and safety tests
├── Dockerfile                          # CPU-only backend image
├── docker-compose.yml                  # Backend service and model cache volume
├── requirements.txt                    # Backend runtime dependencies
├── .env.example                        # Safe configuration template
└── .dockerignore                       # Docker build-context exclusions
```

## Intent and Source Policies

The planner selects a source policy before retrieval. Protected intents do not use unrestricted semantic search or generic web fallback.

| User intent | Policy | Retrieval behavior |
|---|---|---|
| Timetable or timetable slot | `approved_timetable_only` | Requires approved department, semester, section, and 2026–27 records |
| Student list | `approved_student_list_only` | Uses approved department, semester, section, and academic-year metadata |
| R23 regulation | `approved_regulation_pdfs` | Returns approved regulation PDF URL records |
| R23 syllabus | `approved_academic_syllabus_only` | Returns the exact approved syllabus PDF URL |
| Examination results | `approved_exam_results_only` | Returns the approved official result resource |
| Transportation | `approved_transportation_only` | Uses official route and fare records only |
| Multiple facilities | `approved_facilities_only` | Combines approved library, hostel, cafeteria, and transportation evidence |
| Current HOD or principal | `approved_contact_pages` | Uses official contact evidence and protects current-role claims |
| Historical role | `historical_evidence_only` | Allows former-role evidence only for explicitly historical questions |
| General LBRCE question | `pinecone_then_approved_web` | Uses broad retrieval followed by controlled fallback |

## URL-First Academic PDFs

Regulation and syllabus PDFs are intentionally represented as URL-first records. The system stores the official URL and searchable metadata instead of embedding the entire PDF body. When a user asks for one of these documents, the assistant provides the exact official PDF URL and instructs the user to open it for detailed information.

This design avoids unnecessary PDF embedding, avoids consuming hosted embedding quota, and prevents anti-bot HTML pages from being indexed as if they were PDF content.

Typical URL-first questions include:

```text
Show me the official R23 CSE syllabus PDF.

Show me the R23 regulation for M.Tech students.

Where can I find the official examination results page?
```

## Timetable Behavior

The approved timetable scope is **2026–27**. Natural year wording is normalized as follows:

| User wording | Normalized semester |
|---|---|
| 1st year, first year, I year | I |
| 2nd year, second year, II year | III |
| 3rd year, third year, III year | V |
| 4th year, fourth year, IV year | VII |

For example:

```text
Show me the timetable for 3rd year CSE F section.
```

is normalized to:

```text
department: cse
semester: V
section: F
academic_year: 2026-27
```

Approved timetable images are returned as image resources, and approved timetable PDFs are returned as PDF resources. If schedule cells are not safely extractable as text, the assistant identifies the official image or PDF as the authoritative source instead of inventing period values.

## Transportation Behavior

The official [LBRCE transportation page](https://www.lbrce.ac.in/studentcorner_pages/transportation.php) is parsed into route-specific records. Each route vector can store its route code, route points, fee, start time, and official source URL.

Transportation questions are protected from generic web fallback so an admission tuition table cannot be mistaken for a bus-fare table. If the requested location or route is not present in the approved evidence, the assistant should clearly say that the exact route was not found and provide the official source page.

The focused route ingestion command is:

```powershell
python scripts\ingest_transportation_routes_local_bge.py `
  --output-dir migration_artifacts\transportation_routes_local_bge
```

After reviewing the dry-run route count and source URL, write the route vectors with:

```powershell
python scripts\ingest_transportation_routes_local_bge.py `
  --output-dir migration_artifacts\transportation_routes_local_bge `
  --confirm-transportation-migration
```

## Environment Configuration

Create a private `.env` file in the repository root. Never commit it.

```env
# Answer generation
LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=openai/gpt-oss-20b

# Dedicated version-2 Pinecone configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=lbrce-local-bge-index
PINECONE_NAMESPACE=lbrce_local_bge_full_v1

# Local BGE embeddings
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024

# Approved web fallback and official source
TAVILY_API_KEY=your_tavily_api_key
LBRCE_BASE_URL=https://www.lbrce.ac.in
RAG_RELEVANCE_THRESHOLD=0.30

# Frontend origins allowed by the backend
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
```

The frontend contains only a public backend URL. It must never contain Pinecone, Groq, Tavily, or other private API keys.

## Local Development Without Docker

### Backend

From the repository root, create a Python environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the backend:

```powershell
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Check the health endpoint:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

### Frontend

Open another terminal:

```powershell
cd frontend
npm install
```

Create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Start the frontend:

```powershell
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

For a different backend location, change only the public URL:

```env
NEXT_PUBLIC_API_BASE_URL=https://your-backend-url.example.com
```

Never place private backend credentials in the frontend environment.

## Docker Compose

Docker Compose provides a reproducible backend environment with Python 3.11, CPU-only PyTorch, local BGE support, and a persistent Hugging Face model-cache volume.

Keep `.env` beside `Dockerfile` and `docker-compose.yml`, then run:

```powershell
docker compose build --no-cache
docker compose up -d
```

Inspect logs and health:

```powershell
docker compose logs -f lbrce-backend
Invoke-RestMethod http://localhost:8000/health
```

The first startup may download `BAAI/bge-large-en-v1.5`. The named `lbrce_huggingface_cache` volume prevents downloading the model again after every restart.

Stop the service:

```powershell
docker compose down
```

The image is larger than a basic FastAPI image because PyTorch, Transformers, Sentence Transformers, and the BGE runtime are required for compatible local query embeddings.

## Ingestion Workflows

Ingestion scripts should be run selectively. Review dry-run artifacts before any Pinecone write.

### HTML registry

```powershell
python scripts\migrate_html_registry_local_bge.py --help
```

### Student lists

```powershell
python scripts\ingest_student_lists.py `
  --manifest scripts\selected_student_list_pages.json `
  --dry-run
```

### Timetables

```powershell
python scripts\run_ingestion.py `
  --selected-resources scripts\timetable_ingestion_registry.json `
  --timetables-only
```

### Academic URL records

Regulation and syllabus PDFs marked `url_only=true` should be handled by their focused migration scripts. They should not be sent through a generic full-PDF embedding phase.

### Regulation metadata backfill

Existing URL-only regulation records can be checked and patched without re-embedding:

```powershell
python scripts\backfill_regulation_metadata.py
python scripts\backfill_regulation_metadata.py --confirm-backfill
```

The backfill updates Pinecone metadata only. It does not change vector values or download PDF content.

## API Examples

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Chat request:

```powershell
$body = @{ query = "Show me the official R23 CSE syllabus PDF" } | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://localhost:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

A successful response contains an answer, sources, visual resources, and an optional error field:

```json
{
  "answer": "I found the official R23 CSE syllabus PDF. Open it to view the detailed syllabus.",
  "sources": [],
  "visual_resources": [
    {
      "title": "LBRCE R23 CSE Syllabus",
      "url": "https://www.lbrce.ac.in/academics/syllabus/R23/R23_CSE_Syllabus.pdf",
      "type": "pdf"
    }
  ],
  "error": null
}
```

## Recommended End-User Tests

```text
Where is LBRCE located?

What courses are available at LBRCE?

Show me the official R23 CSE syllabus PDF.

Show me the R23 regulation for M.Tech students.

Show me the CSE V semester Section F timetable for 2026–27.

Which students are enrolled in CSE fifth-semester Section F?

Who is the current HOD of CSE?

Who was the former HOD of CSE?

I am living in Singh Nagar. Can you find the nearest bus route for me?

Tell me about the S02 bus route.

I want information about buses, library timing, and hostel facilities.

What is the official website of LBRCE?
```

General questions should not display unrelated syllabus or timetable cards. Explicit document requests should return the exact official PDF URL. Transportation questions should never substitute admission tuition fees for bus fares.

## Testing

Compile the primary backend modules:

```powershell
python -m py_compile backend\main.py backend\api\routes\chat.py backend\graph\nodes.py backend\graph\graph.py backend\graph\state.py backend\retrieval\rag.py
```

Run the focused tests:

```powershell
pytest -q tests\test_final_bugfixes.py
```

Run the complete test suite:

```powershell
pytest -q tests
```

The regression tests cover planner classification, metadata filters, URL-first academic resources, transportation routing, multi-topic facilities, role safeguards, visual-resource filtering, and Pinecone-safe metadata.

## Security

Never commit `.env`, `.env.local`, API keys, Pinecone credentials, Groq keys, Tavily keys, generated archives, model caches, or private ingestion artifacts. The root `.gitignore` is intended to keep these files out of version control.

If a credential has been exposed in a screenshot, chat, log, or public repository, rotate it at the provider before using the project again. Adding a secret to `.gitignore` prevents future commits but does not remove a previously committed secret from Git history.

## Known Limitations

The local BGE model increases the Docker image and memory footprint. This is required for compatibility with the 1024-dimensional local-BGE Pinecone namespace. A smaller embedding model cannot be substituted without creating a new compatible index and re-embedding the corpus.

Some timetable resources are images or scanned PDFs. In those cases, the assistant can display the official visual resource even when it cannot reliably extract individual timetable cells as text.

Regulation and syllabus resources are URL-first by design. The assistant returns the official PDF link and directs the user to open it instead of reproducing the entire document in the answer.

Each question is processed independently. The backend does not retain conversational history between separate requests.

## License and Academic Use

Add a license that matches the project owner’s and institution’s requirements before publishing the repository. If this project is used as an academic demonstration, include appropriate attribution for LBRCE and its official source website.

## References

[1]: [Official LBRCE Website](https://www.lbrce.ac.in/)

[2]: [FastAPI Documentation](https://fastapi.tiangolo.com/)

[3]: [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

[4]: [Pinecone Documentation](https://docs.pinecone.io/)

[5]: [Groq Documentation](https://console.groq.com/docs)

[6]: [Tavily Documentation](https://docs.tavily.com/)

[7]: [Docker Documentation](https://docs.docker.com/)

[8]: [Next.js Documentation](https://nextjs.org/docs)
