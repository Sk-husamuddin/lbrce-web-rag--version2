"""
Real RAG integration test.
Tests two queries against the live Pinecone index + OpenRouter LLM.
Does NOT print API keys.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.config.settings import settings
from backend.embedding.embedding_generator import EmbeddingGenerator
from backend.indexing.pinecone_indexer import PineconeIndexer
from backend.retrieval.rag import RAGPipeline

embedder = EmbeddingGenerator(api_key=settings.PINECONE_API_KEY)
indexer  = PineconeIndexer(
    api_key=settings.PINECONE_API_KEY,
    environment="",
    index_name=settings.PINECONE_INDEX_NAME,
)
pipeline = RAGPipeline(
    embedding_generator=embedder,
    pinecone_indexer=indexer,
    api_key=settings.OPENROUTER_API_KEY,
    base_url=settings.OPENROUTER_BASE_URL,
    model_name=settings.OPENROUTER_MODEL,
)

queries = [
    "What departments are available at LBRCE?",
    "What is the fee structure for Mars University?",   # should get insufficient-evidence response
]

for q in queries:
    print(f"\n{'='*60}")
    print(f"Query: {q}")
    print("="*60)
    result = pipeline.generate_answer(q)
    print(f"Answer:\n{result.get('answer', result)}")
    sources = result.get('sources', [])
    if sources:
        print(f"\nSources ({len(sources)}):")
        for s in sources[:3]:
            print(f"  - {s.get('url', s)}")
    else:
        print("Sources: none")
    if result.get('error'):
        print(f"[ERROR]: {result['error']}")
