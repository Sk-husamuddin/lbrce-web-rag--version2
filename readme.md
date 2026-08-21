# LBRCE Reference Desk — Version 2

A grounded question-answering assistant for **Lakireddy Bali Reddy College of Engineering (LBRCE)**. This version-2 repository is an independent local-development implementation built with FastAPI, LangGraph, local BGE embeddings, Pinecone, Groq-compatible answer generation, controlled Tavily fallback, Docker Compose, and a Next.js frontend.

The project is intended for local development, Docker-based execution, academic demonstration, and continued retrieval-augmented generation experimentation. It uses a dedicated local-BGE Pinecone configuration and does not depend on another deployment.

> The assistant should prefer an honest no-match response over inventing information, mixing unrelated documents, or presenting historical information as current.

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Technology stack](#technology-stack)
- [Embedding and Pinecone configuration](#embedding-and-pinecone-configuration)
- [Repository structure](#repository-structure)
- [Request lifecycle](#request-lifecycle)
- [Intent and source policies](#intent-and-source-policies)
- [Transportation routing](#transportation-routing)
- [Timetables and student lists](#timetables-and-student-lists)
- [URL-first academic resources](#url-first-academic-resources)
- [Environment configuration](#environment-configuration)
- [Local Windows setup](#local-windows-setup)
- [Docker Compose](#docker-compose)
- [Ingestion workflows](#ingestion-workflows)
- [API examples](#api-examples)
- [CLI retrieval debugging](#cli-retrieval-debugging)
- [Testing](#testing)
- [Security](#security)
- [Known limitations](#known-limitations)

## Features

The assistant can answer questions about official LBRCE webpages, departments, admissions, facilities, placements, student lists, timetables, transportation, examination resources, R23 regulations, and R23 syllabus documents.

The application supports the following capabilities:

| Capability | Behavior |
|---|---|
| Local embeddings | Uses `BAAI/bge-large-en-v1.5` locally for both ingestion and query embeddings |
| Metadata filtering | Restricts protected topics to approved Pinecone categories and topics |
| Grounded generation | Sends only selected evidence to the answer-generation model |
| URL-first academic documents | Returns exact official regulation, syllabus, and examination-resource URLs |
| Timetable resources | Returns approved timetable images and PDFs as visual resources |
| Student lists | Supports department, year, semester, section, and academic-year filtering |
| Transportation routes | Uses route-specific records, route points, fees, and start times |
| Typo-tolerant locations | Handles common transliteration differences such as `VELGALERU`, `VELAGALERU`, and `VEELAGALERU` |
| Multi-topic facility retrieval | Preserves library, hostel, and transportation representation while limiting LLM context size |
| CLI retrieval logging | Prints Pinecone score, ID, URL, metadata, and full text for every raw match |
| Stateless requests | Does not retain conversation history between separate requests |

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
   Query planner       Local BGE query       Metadata filters
   and policies         embedding model       and safety guards
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
       Pinecone evidence path       Controlled fallback path
                                      Tavily or approved pages
               |                             |
               +--------------+--------------+
                              v
                    Context assembly
                              |
                              v
                  Deterministic answer paths
                 or Groq answer generation
                              |
                              v
                  Answer, sources, and resources
```

The frontend sends a user question to `POST /chat`. The backend plans the request, retrieves approved evidence, evaluates whether the evidence is sufficient, builds the answer context, and returns a final answer with citations or visual resources.

The raw Pinecone matches are logged in the backend CLI for debugging. They are not returned directly to the browser. The frontend receives the final answer, source cards, visual-resource cards, and an optional error field.

## Technology stack

| Layer | Technology | Role |
|---|---|---|
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS | User interface and resource display |
| API | FastAPI, Uvicorn, Pydantic | HTTP service and request validation |
| Orchestration | LangGraph | Planner, retrieval, evaluation, generation, and retry flow |
| Embeddings | Sentence Transformers with `BAAI/bge-large-en-v1.5` | Local 1024-dimensional document and query vectors |
| Vector search | Pinecone | Similarity search, namespace isolation, and metadata filtering |
| Answer generation | Groq OpenAI-compatible API | Grounded natural-language responses |
| Controlled fallback | Tavily | Restricted official-domain web fallback where permitted |
| Packaging | Docker and Docker Compose | Reproducible CPU-only backend execution |
| Source material | Official LBRCE website | Approved webpages, PDFs, images, tables, and route records |

## Embedding and Pinecone configuration

Version 2 uses a dedicated Pinecone index and namespace:

```env
PINECONE_INDEX_NAME=lbrce-local-bge-index
PINECONE_NAMESPACE=lbrce_local_bge_full_v1
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024
```

The local BGE model is used for both document ingestion and user-query embeddings. Query embeddings use the BGE search instruction prefix:

```text
Represent this sentence for searching relevant passages: 
```

Passage embeddings do not use that query prefix. This asymmetric behavior is required by the BGE model’s intended query-document retrieval format.

Pinecone hosted embedding inference is not used by the local-BGE migration path. Document vectors are generated locally before upsert, and user questions are embedded temporarily for retrieval rather than stored as new document vectors.

### Pinecone index versus namespace

The **index** is the Pinecone vector database with a fixed dimension and similarity metric. The **namespace** is a logical partition inside that index. This project uses the namespace `lbrce_local_bge_full_v1` for the complete local-BGE corpus so it remains separate from benchmark or older migration namespaces.

## Repository structure

```text
.
├── backend/
│   ├── main.py                         # FastAPI application and health route
│   ├── api/routes/chat.py              # Chat request and response contract
│   ├── config/settings.py              # Environment-backed settings
│   ├── graph/
│   │   ├── graph.py                    # LangGraph topology
│   │   ├── state.py                    # Per-request GraphState
│   │   ├── nodes.py                    # Planning, retrieval, policies, answers
│   │   └── constants.py                # Shared fallback constants
│   ├── embedding/
│   │   ├── __init__.py                 # Configured embedding factory
│   │   └── local_bge_embedding_generator.py
│   ├── retrieval/
│   │   ├── __init__.py                 # Pinecone retrieval and result formatting
│   │   └── rag.py                      # Prompt construction and legacy RAG path
│   ├── indexing/
│   │   └── pinecone_indexer.py         # Pinecone connection helpers
│   └── ingestion/
│       ├── html_parser.py              # HTML extraction
│       ├── chunker.py                  # Chunk creation and metadata preservation
│       └── selected_resources.py       # Approved-resource handling
├── frontend/
│   ├── app/                            # Next.js routes and layout
│   ├── components/                     # Chat, sources, and visual resources
│   ├── lib/api.ts                      # Browser API client
│   └── package.json                    # Frontend dependencies and scripts
├── scripts/
│   ├── run_ingestion.py                # General and selective ingestion
│   ├── migrate_html_registry_local_bge.py
│   ├── migrate_academic_resources_local_bge.py
│   ├── migrate_student_corner_local_bge.py
│   ├── ingest_student_lists.py
│   ├── ingest_transportation_routes_local_bge.py
│   ├── backfill_regulation_metadata.py
│   ├── r23_regulations_manifest.json
│   ├── academic_resources_manifest.json
│   └── timetable_ingestion_registry.json
├── tests/                              # Regression and routing tests
├── Dockerfile                          # CPU-only backend image
├── docker-compose.yml                  # Backend and model-cache volume
├── requirements.txt                    # Backend runtime dependencies
├── .env.example                        # Safe configuration template
└── .dockerignore                       # Docker build-context exclusions
```

## Request lifecycle

The request path is implemented primarily in `backend/graph/nodes.py` and `backend/graph/graph.py`:

1. The planner identifies the intent and extracts constraints such as department, year, semester, section, route code, academic year, document type, or topic.
2. The query-rewrite step may improve only general retrieval queries. Protected intents use their planned retrieval query and filters.
3. The retrieval node creates a local BGE query embedding and queries Pinecone in the configured namespace.
4. Protected topics use exact metadata filters such as `page_category` and `topic`.
5. Evidence evaluation applies score, lexical, role-safety, and exact-filter safeguards.
6. Context assembly removes unrelated records and builds the prompt context.
7. Deterministic answer functions handle current HODs, student lists, transportation, timetable resources, and URL-first academic PDFs where deterministic behavior is safer.
8. Other grounded questions use the configured Groq-compatible answer model.
9. The groundedness check prevents unsupported answers and controls whether an approved fallback is permitted.
10. FastAPI returns `answer`, `sources`, `visual_resources`, and an optional `error` field.

## Intent and source policies

The planner selects a source policy before retrieval. Protected intents do not use unrestricted semantic search or arbitrary web fallback.

| User intent | Policy | Retrieval behavior |
|---|---|---|
| Timetable or timetable slot | `approved_timetable_only` | Requires approved department, semester, section, and 2026–27 records |
| Student list | `approved_student_list_only` | Uses approved department, semester, section, and academic-year metadata |
| R23 regulation | `approved_regulation_pdfs` | Returns approved regulation PDF URL records |
| R23 syllabus | `approved_academic_syllabus_only` | Returns the exact approved syllabus PDF URL |
| Examination results | `approved_exam_results_only` | Returns the approved official result resource |
| Transportation | `approved_transportation_only` | Uses official route and fare records only |
| Multiple facilities | `approved_facilities_only` | Combines approved library, hostel, and transportation evidence |
| Current HOD or principal | `approved_contact_pages` | Uses official contact evidence and protects current-role claims |
| Historical role | `historical_evidence_only` | Allows former-role evidence only for explicitly historical questions |
| General LBRCE question | `pinecone_then_approved_web` | Uses broad retrieval followed by controlled fallback |

## Transportation routing

The official [LBRCE transportation page](https://www.lbrce.ac.in/studentcorner_pages/transportation.php) is parsed into route-specific local-BGE records. Each route record can contain a route code, route points, bus fee, start time, academic year, and official source URL.

Transportation questions are protected from generic web fallback so an admission-fee table cannot be mistaken for a bus-fare table. The deterministic transportation path supports the following behavior:

| Question type | Behavior |
|---|---|
| Explicit route code, such as `S02` or `J16` | Extracts route rows and displays a Markdown table when available |
| Location question, such as Mylavaram | Searches the retrieved route documents and can return multiple matching route codes |
| Transliteration variant, such as `VELGALERU` | Uses word-level fuzzy matching with a similarity cutoff of `0.78` |
| Multiple route matches | Uses a generic transportation citation title rather than naming only one route |
| No approved match | Returns an honest no-match message and the official transportation page |

The route-specific ingestion script is:

```powershell
python scripts\ingest_transportation_routes_local_bge.py `
  --output-dir migration_artifacts\transportation_routes_local_bge `
  --namespace lbrce_local_bge_full_v1
```

Review the dry-run output before writing vectors. When the route count and source URL are correct, explicitly confirm the migration:

```powershell
python scripts\ingest_transportation_routes_local_bge.py `
  --output-dir migration_artifacts\transportation_routes_local_bge `
  --namespace lbrce_local_bge_full_v1 `
  --confirm-transportation-migration
```

## Timetables and student lists

The approved timetable scope is **2026–27**. Natural year wording is normalized as follows:

| User wording | Normalized semester |
|---|---|
| 1st year, first year, I year | I |
| 2nd year, second year, II year | III |
| 3rd year, third year, III year | V |
| 4th year, fourth year, IV year | VII |

For example, the following request is normalized to CSE, semester V, section F, academic year 2026–27:

```text
Show me the timetable for 3rd year CSE F section.
```

Approved timetable images are returned as image resources, and approved timetable PDFs are returned as PDF resources. When individual timetable cells cannot be extracted reliably, the assistant identifies the official image or PDF instead of inventing period values.

Student-list records use approved department, semester, section, academic-year, and resource-type metadata. Student-list requests do not use generic Tavily fallback when an exact approved roster is unavailable.

## URL-first academic resources

Regulation, syllabus, and examination PDF resources are represented as URL-first records. The system stores the official URL and searchable metadata instead of embedding the entire PDF body. When a user asks for one of these documents, the assistant provides the exact official PDF URL and directs the user to open it for detailed information.

This avoids unnecessary PDF embedding, prevents blocked or textless PDF responses from becoming misleading evidence, and avoids using hosted embedding inference for document migration.

Typical requests include:

```text
Show me the official R23 CSE syllabus PDF.

Show me the R23 regulation for M.Tech students.

Where can I find the official examination results page?
```

## Environment configuration

Create a private `.env` file in the repository root. Never commit it and never place private credentials in the frontend.

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

# Backend CORS origins
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]
```

The frontend should contain only the public backend URL:

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

The frontend must not contain Pinecone, Groq, Tavily, or other private API keys.

## Local Windows setup

### Backend without Docker

From PowerShell, open the repository root and create a virtual environment:

```powershell
cd "C:\lbrce-web-rag -version2"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
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

Open a second PowerShell window:

```powershell
cd "C:\lbrce-web-rag -version2\frontend"
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

## Docker Compose

Docker Compose provides a reproducible backend environment with Python 3.11, CPU-only PyTorch, local BGE support, and a persistent Hugging Face model-cache volume.

Keep `.env` beside `Dockerfile` and `docker-compose.yml`, then run:

```powershell
cd "C:\lbrce-web-rag -version2"
docker compose build --no-cache
docker compose up -d
```

Inspect logs and health:

```powershell
docker compose logs -f lbrce-backend
Invoke-RestMethod http://localhost:8000/health
```

After replacing a backend Python file, restart the service. A rebuild is safest when the file is copied into the image:

```powershell
docker compose down
docker compose build --no-cache
docker compose up -d
```

The first startup may download `BAAI/bge-large-en-v1.5`. The named Hugging Face cache volume prevents downloading the model again after every restart.

The image is larger than a basic FastAPI image because PyTorch, Transformers, Sentence Transformers, and the BGE runtime are required for compatible local query embeddings.

Stop the service with:

```powershell
docker compose down
```

## Ingestion workflows

Ingestion scripts should be run selectively. Review dry-run artifacts before any Pinecone write. Local BGE ingestion must not be replaced with Pinecone hosted embedding calls.

### Full HTML registry

```powershell
python scripts\migrate_html_registry_local_bge.py --help
```

Use the project’s approved registry and review the generated audit files before uploading vectors. The audit should report successful extraction, metadata assignment, and `pinecone_written: false` during preparation.

### Student lists

```powershell
python scripts\ingest_student_lists.py `
  --manifest scripts\selected_student_list_pages.json `
  --dry-run
```

After reviewing the chunks and metadata, run the confirmed ingestion command documented by the script’s `--help` output. Student-list vectors should use the approved resource type and department/year metadata.

### Timetables

```powershell
python scripts\run_ingestion.py `
  --selected-resources scripts\timetable_ingestion_registry.json `
  --timetables-only
```

This selected-resource phase stores timetable image and PDF metadata without requiring the full website crawl.

### Student Corner pages

```powershell
python scripts\migrate_student_corner_local_bge.py --help
```

The student-corner migration covers approved pages such as the bank, library, hostel, cafeteria, clubs, internet, sports, dispensary, and transportation directory records.

### Academic URL records

Regulation and syllabus PDFs marked `url_only=true` should be handled by the focused academic migration scripts. They should not be sent through a generic full-PDF embedding phase.

### Regulation metadata backfill

Existing URL-only regulation records can be checked and patched without re-embedding:

```powershell
python scripts\backfill_regulation_metadata.py
python scripts\backfill_regulation_metadata.py --confirm-backfill
```

The backfill updates Pinecone metadata only. It does not change vector values or download PDF content.

## API examples

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

Raw Pinecone match metadata and full text are intentionally not included in this public response contract. They are available in the backend CLI logs for debugging.

## CLI retrieval debugging

After every Pinecone query, `backend/graph/nodes.py` logs each raw match in the backend terminal:

```text
[PINECONE MATCH 1]
score: 0.812
id: transport-routes-bge-...
source_url: https://www.lbrce.ac.in/studentcorner_pages/transportation.php
metadata: {...}
full_text:
...
```

The log includes the score, vector ID, source URL, metadata, and complete retrieved text. This logging does not call an LLM, rewrite the question, modify Pinecone, or expose the raw records in the frontend.

For multi-topic facility queries, all raw candidates are logged first. The context sent to Groq is then limited to at most three highest-scoring chunks per topic:

```text
[retrieve_node] facility_multi_topic trimmed 43 raw chunks to 9 for context (cap=3 per topic).
```

This prevents near-duplicate transportation routes from overflowing the answer-generation payload while preserving library, hostel, and transportation representation.

## Recommended end-user tests

Run these questions from the frontend after starting both services:

```text
Where is LBRCE located?

What courses are available at LBRCE?

Show me the official R23 CSE syllabus PDF.

Show me the R23 regulation for M.Tech students.

Show me the CSE V semester Section F timetable for 2026–27.

Which students are enrolled in CSE fifth-semester Section F?

Who is the current HOD of CSE?

Who was the former HOD of CSE?

Tell me about route S02.

Which route serves Mylavaram?

Tell me which bus travel via VELGALERU?

Tell which bus routes will travel through JAKAMPUDI?

Tell me which bus travels through Hyderabad?

Is there a bank on the LBRCE campus?

Tell me about the buses, library, and hostel facilities at LBRCE.
```

Expected transportation behavior includes route rows for explicit route-code queries, J16/J19 for Mylavaram, transliteration-tolerant matching for VELGALERU, no route for unrelated Hyderabad, and a citation title consistent with the route or route set named in the answer.

General questions should not display unrelated syllabus or timetable cards. Explicit document requests should return the exact official PDF URL. Transportation questions should never substitute admission tuition fees for bus fares.

## Testing

Compile the primary backend modules:

```powershell
python -m py_compile `
  backend\main.py `
  backend\api\routes\chat.py `
  backend\graph\nodes.py `
  backend\graph\graph.py `
  backend\graph\state.py `
  backend\retrieval\rag.py
```

Run the focused regression tests:

```powershell
pytest -q tests\test_final_bugfixes.py
```

Run the complete test suite:

```powershell
pytest -q tests
```

The regression suite covers planner classification, metadata filters, URL-first academic resources, transportation routing, fuzzy location matching, multi-topic facilities, role safeguards, visual-resource filtering, and Pinecone-safe metadata.

## Security

Never commit `.env`, `.env.local`, API keys, Pinecone credentials, Groq keys, Tavily keys, generated archives, model caches, or private ingestion artifacts.

If a credential has been exposed in a screenshot, chat, log, or repository, rotate it at the provider before using the project again. Adding a secret to `.gitignore` prevents future commits but does not remove a previously committed secret from Git history.

The frontend must contain only public configuration such as `NEXT_PUBLIC_API_BASE_URL`. All service credentials belong in the backend environment or deployment secret store.

## Known limitations

The local BGE model increases the Docker image size and memory footprint. It is required for compatibility with the 1024-dimensional local-BGE Pinecone namespace. A different embedding model cannot be substituted without creating a compatible index and re-embedding the corpus.

Some timetable resources are images or scanned PDFs. In those cases, the assistant can display the official visual resource even when it cannot reliably extract individual timetable cells as text.

Regulation and syllabus resources are URL-first by design. The assistant returns the official PDF link and directs the user to open it instead of reproducing the entire document in the answer.

A multi-topic request is limited to a representative number of context chunks per topic to stay within the answer-generation provider’s request-size limits. The complete raw Pinecone candidate set remains available in the backend CLI logs.

Each question is processed independently. The backend does not retain conversational history between separate requests.

## Academic use and attribution

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

[9]: [Sentence Transformers Documentation](https://www.sbert.net/)
