# LBRCE Reference Desk

A grounded question-answering assistant for **Lakireddy Bali Reddy College of Engineering (LBRCE)**. The system combines FastAPI, LangGraph, Pinecone retrieval, Groq answer generation, Tavily fallback search, and a Next.js frontend to answer questions about LBRCE webpages, departments, regulations, timetables, and official visual resources.

> The assistant is designed to answer from approved LBRCE evidence. It should prefer an honest no-match response over an unsupported or historical answer presented as current.

## Project Overview

LBRCE Reference Desk is a retrieval-augmented generation application. The backend retrieves relevant indexed evidence, applies intent-specific filtering, assembles a grounded context, and asks the configured language model to generate the final answer. Timetable images and PDFs are returned as visual resources so users can inspect the official schedule directly.

The project is deployed as two services:

| Layer | Technology | Deployment |
| --- | --- | --- |
| Frontend | Next.js, React, TypeScript, Tailwind CSS | Vercel |
| Backend | FastAPI, LangGraph, Python | Render |
| Vector search | Pinecone | Pinecone Cloud |
| Answer generation | Groq OpenAI-compatible chat API | Groq |
| Controlled web fallback | Tavily | Tavily API |
| Official resources | LBRCE webpages, PDFs, timetable images, and timetable PDFs | LBRCE website |

## Current Architecture

The LangGraph workflow uses a deterministic planner before Pinecone retrieval:

```
User question
    |
    v
plan_query_node
    |
    |-- intent
    |-- normalized_query
    |-- retrieval_query
    |-- query_filters
    |-- source_policy
    |-- planner_confidence
    v
retrieve_node
    |
    v
 evaluate_evidence_node
    |
    |-- sufficient evidence --> assemble_context_node
    |
    |-- insufficient evidence --> controlled fallback strategy
                                      |
                                      v
                              Tavily or official contact retrieval
                                      |
                                      v
                              assemble_context_node
                                      |
                                      v
generate_answer_node
    |
    v
check_groundedness_node
    |
    v
Final answer, sources, and visual resources
```

The planner does not answer questions and does not create vectors. It normalizes the user’s intent and constraints before the single Pinecone query embedding is generated. This prevents different downstream functions from interpreting the same question in different ways.

### Planner source policies

| Intent | Source policy | Purpose |
| --- | --- | --- |
| `timetable` | `approved_timetable_only` | Match only approved timetable images and PDFs |
| `timetable_slot` | `approved_timetable_only` | Handle period or subject questions using the official visual resource |
| `current_department_role` | `approved_contact_pages` | Protect current HOD answers using department contact evidence |
| `current_principal` | `approved_contact_pages` | Prevent historical principal names from being presented as current |
| `aggregate_hod` | `approved_contact_pages` | Collect evidence department by department rather than using one broad search |
| `historical_role` | `historical_evidence_only` | Permit former-role evidence without converting it into current evidence |
| `regulation` | `approved_regulation_pdfs` | Prefer the approved R23 regulation documents |
| `general` | `pinecone_then_approved_web` | Answer ordinary LBRCE webpage questions with controlled fallback |

## Supported Timetable Normalization

The planner understands natural year and semester wording. For the approved B.Tech timetable structure, the following mappings are applied:

| User wording | Normalized semester |
| --- | --- |
| 1st year, first year, I year | I |
| 2nd year, second year, II year | III |
| 3rd year, third year, III year | V |
| 4th year, fourth year, IV year | VII |

The approved timetable scope is **2026–27**. If a timetable question omits the academic year, the planner applies `2026-27` as the default. Department, semester, section, academic year, and time-range constraints are then used for strict filtering.

For example:

```
Show me the timetable for 3rd year CSE F section.
```

is normalized to:

```
department: cse
semester: V
section: F
academic_year: 2026-27
```

The retrieval query is expanded with official LBRCE timetable wording to improve recall, while the original question remains available for filtering and answer generation.

## Timetable and Visual Resource Behavior

Approved timetable images are returned as frontend image resources. Approved timetable PDFs are returned as frontend PDF resources. The system does not claim individual timetable cell values unless those values are present in textual evidence.

If a timetable resource is found but its schedule cells are not extractable as text, the assistant explains that the official image or PDF is the authoritative source and displays the resource below the answer.

If no approved timetable matches the requested department, semester, section, and academic year, the system returns a safe no-match response. It does not use generic Tavily search to retrieve unrelated syllabus PDFs, course-structure documents, or examination schedules.

## LBRCE Resource Scope

The current approved corpus includes the following categories:

| Resource category | Coverage |
| --- | --- |
| LBRCE webpages | Indexed college, department, course, admissions, facilities, and contact pages |
| Department contact pages | Approved official contact pages for department-aware HOD retrieval |
| R23 regulation PDFs | Selected official R23 B.Tech, M.Tech, MBA, and Honors / Minors documents |
| R23 academic PDFs | Selected course-structure and regulation resources |
| 2026–27 timetable images | Approved department-wise timetable records, including CSE section images |
| 2026–27 timetable PDFs | Approved semester-level timetable PDFs for departments whose official pages provide PDFs |

Timetable pixels are treated as URL-based visual resources. They are not required to be OCR-processed for the frontend to display them.

## Role and Principal Safeguards

Current HOD and principal questions are protected against stale evidence. Former, ex-, previous, past, and historical names are not presented as current office holders.

Aggregate questions such as the following use the approved official department-contact directory:

```
List out the departments along with the HOD's names for LBRCE.
```

The backend recognizes natural variants such as `department-wise HOD names`, `all departments with their heads`, and `list every department and its current HOD`. Each department is evaluated independently. If a department’s official page does not explicitly confirm a current HOD, it should be reported as unconfirmed rather than inferred from another page.

Historical questions are handled separately:

```
Who was the former HOD of CSE?
```

Historical evidence may answer an explicitly historical question, but the same evidence cannot support a current-role answer.

## Technology Stack

The backend uses Python 3.10 or later, FastAPI for HTTP endpoints, LangGraph for orchestration, Pinecone for vector search, and an OpenAI-compatible client for Groq or OpenRouter answer generation. The frontend uses Next.js, React, TypeScript, and Tailwind CSS.

Relevant backend files include:

```
backend/
├── main.py                         # FastAPI application and CORS
├── config/settings.py              # Environment-backed configuration
├── graph/
│   ├── graph.py                    # LangGraph topology
│   ├── state.py                    # GraphState fields
│   ├── nodes.py                    # Planner, retrieval, routing, filtering, and generation
│   └── constants.py                # Shared fallback answer constant
├── retrieval/
│   └── rag.py                      # Legacy RAGPipeline prompt and retrieval path
├── indexing/
│   └── pinecone_indexer.py         # Pinecone connection and upsert helpers
├── embedding/
│   └── embedding_generator.py      # Query/document embedding interface
└── ingestion/
    ├── html_parser.py              # HTML extraction
    ├── chunker.py                  # Chunk creation and metadata preservation
    └── selected_resources.py       # Approved PDF and timetable metadata chunks

frontend/
├── app/                            # Next.js routes and layout
├── components/                     # Reference Desk, chat, sources, and resources
└── lib/api.ts                      # Backend API client

scripts/
├── run_ingestion.py                # Selective and general ingestion entry point
├── timetable_ingestion_registry.json
└── r23_regulations_manifest.json
```

## Local Backend Setup

Clone the repository and create a Python virtual environment:

```
git clone https://github.com/YOUR_USERNAME/lbrce-reference-desk.git
cd lbrce-reference-desk
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a local `.env` file at the repository root. Never commit this file.

```
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=lbrce-index

LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant

TAVILY_API_KEY=your_tavily_key
LBRCE_BASE_URL=https://www.lbrce.ac.in
RAG_RELEVANCE_THRESHOLD=0.30
CORS_ORIGINS=http://localhost:3000
```

Optional OpenRouter settings can be used instead of Groq:

```
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=your_openrouter_model
```

Start the backend from the repository root:

```
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Check the health endpoint:

```
http://localhost:8000/health
```

## Local Frontend Setup

Open another terminal:

```
cd frontend
pnpm install
```

Create `frontend/.env.local`:

```
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

Start the development server:

```
pnpm dev
```

Open:

```
http://localhost:3000
```

Only the public backend URL belongs in the frontend environment. Pinecone, Groq, Tavily, and OpenRouter keys must remain on the backend.

## Selective Ingestion

Selective ingestion is recommended because it prevents accidental crawling or re-embedding of unrelated resources.

### R23 regulation PDFs

Copy the approved manifest into `scripts/r23_regulations_manifest.json`, then run:

```
python scripts/run_ingestion.py --selected-resources scripts\r23_regulations_manifest.json --max-pdfs 4
```

### 2026–27 timetable resources

Use the validated timetable manifest:

```
python scripts/run_ingestion.py --selected-resources scripts\timetable_ingestion_registry.json --timetables-only
```

The timetable workflow creates URL-based metadata chunks for approved timetable PDFs and image records. It does not require OCR or vision processing for frontend display.

Successful ingestion records are checkpointed. Re-running the same selective command should skip records that have already been processed.

Avoid using large ingestion commands during demonstrations. User questions are embedded temporarily for retrieval; they are not chunked, stored, or added to Pinecone as new document vectors.

## API Usage

The backend exposes a health endpoint and a chat endpoint.

Health check:

```
Invoke-RestMethod http://localhost:8000/health
```

Chat request:

```
$body = @{ query = "Show the CSE V semester Section F timetable for 2026-27" } | ConvertTo-Json

Invoke-RestMethod `
  -Uri http://localhost:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

A successful response may contain:

```json
{
  "answer": "I found the matching official LBRCE timetable image. Please view it below.",
  "sources": [],
  "visual_resources": [
    {
      "title": "Computer Science & Engineering V Semester Section F Timetable 2026-27",
      "url": "https://www.lbrce.ac.in/.../V_SEM_F_SEC_TT_AY_2026-27.jpg",
      "type": "image",
      "department": "cse",
      "academic_year": "2026-27",
      "semester": "V",
      "section": "F"
    }
  ],
  "error": null
}
```

## Recommended Test Questions

```
What undergraduate programs does LBRCE offer?
```

```
Show me the timetable for 3rd year CSE F section.
```

```
What is the subject for 2nd year CSE H section at 9 to 10 AM?
```

```
Show the AI and Data Science Section B timetable for 2026-27.
```

```
Who is the current HOD of CSE?
```

```
Who was the former HOD of CSE?
```

```
List out the departments along with the HOD's names for LBRCE.
```

```
What are the R23 B.Tech regulations at LBRCE?
```

```
Show the CSE Section F timetable for 2030-31.
```

The final query is a safety test. It should not return an unrelated timetable or regulation PDF.

## Testing

Run syntax checks:

```
python -m py_compile backend\graph\nodes.py backend\retrieval\rag.py backend\graph\state.py backend\graph\graph.py backend\graph\constants.py
```

Run the focused planner, timetable, role, and prompt tests:

```
pytest -q tests\test_planner_regression.py tests\test_timetable_ai_ds_regression.py tests\test_principal_routing_regression.py tests\test_hod_routing_regression.py tests\test_historical_hod_regression.py tests\test_audit_bugfixes.py
```

Run the complete backend suite:

```
pytest -q tests\
```

The latest targeted validation passed **35 tests** after the historical evidence-field fix. The last complete backend validation passed **71 tests with one known pre-existing crawler test failure**. That test expects `crawl( )` to return two values while the current crawler returns three values. It is unrelated to the planner, retrieval, prompt, timetable, HOD, or principal logic.

## Production Deployment

### Render backend

Create a Render Web Service using the repository root:

| Setting | Value |
| --- | --- |
| Runtime | Python 3 |
| Root directory | Repository root |
| Build command | `pip install -r requirements.txt` |
| Start command | `uvicorn backend.main:app --host 0.0.0.0 --port $PORT` |
| Health check path | `/health` |

Add the backend secrets and configuration variables in Render only:

```
PINECONE_API_KEY=your_real_key
PINECONE_INDEX_NAME=lbrce-index
LLM_PROVIDER=groq
GROQ_API_KEY=your_real_key
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant
TAVILY_API_KEY=your_real_key
LBRCE_BASE_URL=https://www.lbrce.ac.in
RAG_RELEVANCE_THRESHOLD=0.30
CORS_ORIGINS=https://your-frontend.vercel.app
```

### Vercel frontend

Import the repository into Vercel and set the Root Directory to:

```
frontend
```

Use the detected Next.js framework and leave the output directory at its default. Configure:

```
NEXT_PUBLIC_API_BASE_URL=https://your-backend.onrender.com
```

Do not add Pinecone, Groq, Tavily, or OpenRouter keys to Vercel. The `NEXT_PUBLIC_` prefix intentionally exposes the backend URL to browser code, so it must only contain a public URL.

After Vercel deployment, update Render’s CORS variable with the exact production Vercel origin:

```
CORS_ORIGINS=https://lbrce-reference-desk.vercel.app
```

Redeploy Render after changing CORS.

## Security Practices

Never commit `.env`, `.env.local`, API keys, Pinecone credentials, generated archives, `node_modules`, `.next`, virtual environments, or ingestion caches. Use the repository root `.gitignore`.

If a secret was ever committed to a public Git repository, rotate it at the provider immediately. Adding `.env` to `.gitignore` prevents future commits but does not remove a previously committed secret from Git history.

The frontend should contain only public configuration such as:

```
NEXT_PUBLIC_API_BASE_URL=https://your-backend.onrender.com
```

## Known Limitations

The assistant cannot guarantee a timetable period value when the official timetable is available only as an image or a scanned PDF and no OCR text is indexed. In those cases, it presents the official visual resource as the authoritative source.

Historical answers require historical evidence. If the indexed corpus or controlled web search does not contain a reliable former-role statement, the assistant returns a safe no-match response.

Aggregate HOD answers depend on official department contact pages. Departments without an explicit current HOD statement should be marked unconfirmed rather than inferred.

The crawler test contract still needs a separate maintenance update because the current crawler returns three values while one legacy test expects two. This does not affect the deployed RAG path.

## License

Add the license that matches your project and institutional requirements before publishing the repository. If this is an academic demonstration project, include an appropriate attribution and usage statement.

## References

[1]: [FastAPI Documentation](https://fastapi.tiangolo.com/)

[2]: [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

[3]: [Pinecone Documentation](https://docs.pinecone.io/)

[4]: [Groq Documentation](https://console.groq.com/docs)

[5]: [Tavily Documentation](https://docs.tavily.com/)

[6]: [Render Documentation](https://render.com/docs)

[7]: [Vercel Next.js Documentation](https://vercel.com/docs/frameworks/nextjs)
