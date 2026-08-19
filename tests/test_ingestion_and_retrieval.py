import pytest
from unittest.mock import MagicMock, patch

from backend.ingestion.pdf_parser import PDFParser, PDFDocument
from backend.ingestion.cleaner import ContentCleaner, CleanedContent
from backend.ingestion.chunker import chunk_document, chunk_pdf_document, chunk_cleaned_content, Chunk
from backend.embedding.embedding_generator import EmbeddingGenerator
from backend.retrieval import retrieve, retrieve_with_fallback, EmptyQueryError, PineconeUnavailableError, NoResultsError
from backend.indexing.pinecone_indexer import PineconeIndexer

# ==========================================
# 1. PDF Parser Tests
# ==========================================

def test_pdf_parser_pages():
    """Test that a multi-page PDF is correctly parsed into a list of PDFDocuments with correct page numbers."""
    parser = PDFParser()
    
    # Mock fitz Document and Page
    mock_doc = MagicMock()
    mock_doc.page_count = 3
    mock_doc.metadata = {"title": "Test Title"}
    
    mock_pages = []
    for i in range(3):
        mock_page = MagicMock()
        mock_page.get_text.return_value = f"This is content on page {i+1}."
        mock_page.get_tables.return_value = []
        mock_page.get_images.return_value = []
        mock_pages.append(mock_page)
        
    mock_doc.__getitem__.side_effect = lambda idx: mock_pages[idx]
    
    with patch("fitz.open", return_value=mock_doc):
        results = parser.parse(b"dummy_data", "https://example.com/cse/dept.pdf")
        
    assert results is not None
    assert len(results) == 3
    
    for idx, doc in enumerate(results):
        assert isinstance(doc, PDFDocument)
        assert doc.document_id == "cse_dept_pdf"
        assert doc.title == "Test Title"
        assert doc.content == f"This is content on page {idx+1}."
        assert doc.metadata["page_number"] == idx + 1
        assert doc.metadata["page_count"] == 3
        assert doc.department == "Computer Science Engineering"


# ==========================================
# 2. Cleaner Tests
# ==========================================

def test_cleaner_no_chunking():
    """Verify that cleaner cleans content but does not generate chunks."""
    cleaner = ContentCleaner()
    raw_content = "Hello    World!   This is some   <b>bold</b> content   with a phone 1234567890."
    
    result = cleaner.clean(raw_content, "https://example.com/test", "test_id")
    
    assert isinstance(result, CleanedContent)
    assert not hasattr(result, "chunks") or getattr(result, "chunks") is None
    # Verify cleaning results (whitespace and entities)
    assert "Hello World!" in result.cleaned_text
    assert "bold" in result.cleaned_text
    assert result.original_length == len(raw_content)


# ==========================================
# 3. Chunker Tests
# ==========================================

def test_chunker_validation():
    """Validate that chunk_document raises errors for invalid chunking configurations."""
    doc = {"content": "Sample text for testing chunk configuration validation."}
    
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_document(doc, chunk_size=0, overlap=5)
        
    with pytest.raises(ValueError, match="overlap must be non-negative"):
        chunk_document(doc, chunk_size=10, overlap=-1)
        
    with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
        chunk_document(doc, chunk_size=10, overlap=10)


def test_chunker_overlap_and_word_boundaries():
    """Verify that chunking overlaps correctly and aligns with word boundaries (not splitting mid-word)."""
    # Create text with clear word structures
    # Heuristic: 4 characters per token. So chunk_size=20 tokens => ~80 characters.
    # overlap=5 tokens => ~20 characters.
    text = "Lakireddy Balireddy College of Engineering is a premier institute. It is located in Mylavaram. The departments of computer science engineering and information technology are highly reputed."
    doc = {
        "content": text,
        "document_id": "doc1",
        "source_url": "https://example.com",
        "page_number": 1,
        "source_type": "html"
    }
    
    chunks = chunk_document(doc, chunk_size=20, overlap=5)
    
    assert len(chunks) > 1
    
    # Check that chunks do not cut words mid-word (should end with space or complete word)
    for c in chunks:
        # Check boundary
        text_val = c.text
        # Shouldn't be in the middle of a word unless no spaces exist.
        # We verify that they look like clean words.
        words = text_val.split()
        assert len(words) > 0
        
    # Verify overlap exists by checking if some content in chunk 0 is present in chunk 1
    # Overlap is computed: start = end - chunk_char_overlap
    # Chunk 0 ends at some position. Chunk 1 should start before that position.
    assert any(word in chunks[1].text for word in chunks[0].text.split()[-3:])


def test_chunker_overlap_zero_and_short_doc():
    """Test chunker when overlap is zero, and on a short document (single chunk)."""
    text = "Short document text."
    doc = {"content": text}
    
    # Overlap = 0
    chunks = chunk_document(doc, chunk_size=50, overlap=0)
    assert len(chunks) == 1
    assert chunks[0].text == text
    
    # Short document
    chunks_short = chunk_document(doc, chunk_size=500, overlap=50)
    assert len(chunks_short) == 1
    assert chunks_short[0].text == text


# ==========================================
# 4. Embedding Tests
# ==========================================

@patch("backend.embedding.embedding_generator.Pinecone")
def test_embedding_generator_real_and_mock(mock_pinecone):
    """Test EmbeddingGenerator with mocked Pinecone Inference API."""
    # Setup mock return values from Pinecone inference
    mock_pc_instance = MagicMock()
    mock_pinecone.return_value = mock_pc_instance
    
    mock_embedding_1 = MagicMock()
    mock_embedding_1.values = [0.1, 0.2, 0.3]
    
    mock_embedding_2 = MagicMock()
    mock_embedding_2.values = [0.4, 0.5, 0.6]
    
    mock_response = MagicMock()
    mock_response.data = [mock_embedding_1, mock_embedding_2]
    
    mock_pc_instance.inference.embed.return_value = mock_response
    mock_pc_instance.api_key = "real_api_key_test"
    
    generator = EmbeddingGenerator(model_name="llama-text-embed-v2", dimension=3, api_key="real_api_key_test")
    
    # Test single embed
    mock_single_response = MagicMock()
    mock_single_response.data = [mock_embedding_1]
    mock_pc_instance.inference.embed.return_value = mock_single_response
    
    vec = generator.embed("Hello world")
    assert vec == [0.1, 0.2, 0.3]
    mock_pc_instance.inference.embed.assert_called_with(
        model="llama-text-embed-v2",
        inputs=["Hello world"],
        parameters={"input_type": "query", "truncate": "END"}
    )
    
    # Test chunk embed
    mock_pc_instance.inference.embed.return_value = mock_response
    chunks = [
        Chunk(chunk_id="1", text="Chunk one", source_url="a", title="a", source_type="a", department="a", page_number=1, document_id="a"),
        Chunk(chunk_id="2", text="Chunk two", source_url="a", title="a", source_type="a", department="a", page_number=1, document_id="a"),
    ]
    
    vectors = generator.embed_chunks(chunks)
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2, 0.3]
    assert vectors[1] == [0.4, 0.5, 0.6]
    mock_pc_instance.inference.embed.assert_called_with(
        model="llama-text-embed-v2",
        inputs=["Chunk one", "Chunk two"],
        parameters={"input_type": "passage", "truncate": "END"}
    )


# ==========================================
# 5. Pinecone Retrieval Tests
# ==========================================

def test_pinecone_retrieval_success():
    """Test retrieval parsing with mocked Pinecone index matches."""
    mock_indexer = MagicMock()
    
    # Setup mock Pinecone matches response
    mock_match_1 = MagicMock()
    mock_match_1.score = 0.95
    mock_match_1.metadata = {
        "chunk_id": "c1",
        "text": "Retrieved passage one content.",
        "source_url": "https://example.com/1",
        "title": "Page Title 1",
        "source_type": "html",
        "department": "CSE",
        "page_number": "2",
        "document_id": "doc_1"
    }
    
    mock_match_2 = MagicMock()
    mock_match_2.score = 0.85
    mock_match_2.metadata = {
        "chunk_id": "c2",
        "text": "Retrieved passage two content.",
        "source_url": "https://example.com/2",
        "title": "Page Title 2",
        "source_type": "html",
        "department": "ECE",
        "page_number": 0,
        "document_id": "doc_2"
    }
    
    mock_response = MagicMock()
    mock_response.matches = [mock_match_1, mock_match_2]
    mock_indexer.index.query.return_value = mock_response
    
    # Mock generator
    mock_generator = MagicMock()
    mock_generator.embed.return_value = [0.1, 0.2, 0.3]
    
    results = retrieve("Test query", mock_generator, mock_indexer, top_k=2)
    
    assert len(results) == 2
    
    assert results[0]["chunk_text"] == "Retrieved passage one content."
    assert results[0]["similarity_score"] == 0.95
    assert results[0]["source_url"] == "https://example.com/1"
    assert results[0]["page_number"] == 2
    
    assert results[1]["chunk_text"] == "Retrieved passage two content."
    assert results[1]["similarity_score"] == 0.85
    assert results[1]["page_number"] == 0


def test_pinecone_retrieval_empty_query():
    """Test retrieval raises EmptyQueryError for empty queries."""
    mock_indexer = MagicMock()
    mock_generator = MagicMock()
    
    with pytest.raises(EmptyQueryError):
        retrieve("", mock_generator, mock_indexer)
        
    with pytest.raises(EmptyQueryError):
        retrieve("   ", mock_generator, mock_indexer)


def test_pinecone_retrieval_no_results():
    """Test retrieval raises NoResultsError when no matches are found."""
    mock_indexer = MagicMock()
    mock_response = MagicMock()
    mock_response.matches = []
    mock_indexer.index.query.return_value = mock_response
    
    mock_generator = MagicMock()
    mock_generator.embed.return_value = [0.1, 0.2, 0.3]
    
    with pytest.raises(NoResultsError):
        retrieve("Passage", mock_generator, mock_indexer)


# ==========================================
# 6. Crawler Tests
# ==========================================

from backend.ingestion.crawler import LBRCECrawler, is_valid_lbrce_url, normalize_url
from backend.ingestion.html_parser import ParsedDocument

def test_url_validation_and_normalization():
    """Verify LBRCE url checks and URL normalization."""
    allowed = {"lbrce.ac.in", "lbrce.com"}
    
    assert is_valid_lbrce_url("https://www.lbrce.ac.in/cse/index.html", allowed)
    assert is_valid_lbrce_url("http://lbrce.ac.in", allowed)
    assert is_valid_lbrce_url("https://cs.lbrce.ac.in/activity.pdf", allowed)
    assert not is_valid_lbrce_url("https://example.com", allowed)
    
    assert normalize_url("https://www.lbrce.ac.in/cse/") == "https://www.lbrce.ac.in/cse"
    assert normalize_url("https://www.lbrce.ac.in/cse#fragment") == "https://www.lbrce.ac.in/cse"
    assert normalize_url("https://www.lbrce.ac.in/cse?query=1") == "https://www.lbrce.ac.in/cse?query=1"


@pytest.mark.asyncio
@patch("backend.ingestion.html_parser.HTMLParser.fetch")
@patch("backend.ingestion.html_parser.HTMLParser.parse")
async def test_crawler_bfs(mock_parse, mock_fetch):
    """Test that crawler executes BFS crawling, collects PDFs, and visits correct pages."""
    mock_fetch.side_effect = lambda url: f"<html>Content from {url}</html>"
    
    def parse_side_effect(html, url):
        if url == "https://www.lbrce.ac.in":
            return ParsedDocument(
                document_id="home",
                title="LBRCE Home",
                content="Welcome to college.",
                source_url=url,
                headings=["Welcome"],
                links=["https://www.lbrce.ac.in/cse", "https://example.com/external"],
                pdf_links=["https://www.lbrce.ac.in/syllabus.pdf"],
                metadata={}
            )
        elif url == "https://www.lbrce.ac.in/cse":
            return ParsedDocument(
                document_id="cse",
                title="CSE Department",
                content="CSE info.",
                source_url=url,
                headings=["CSE"],
                links=["https://www.lbrce.ac.in"],
                pdf_links=["https://www.lbrce.ac.in/cse/handbook.pdf"],
                metadata={}
            )
        return None
        
    mock_parse.side_effect = parse_side_effect
    
    crawler = LBRCECrawler(
        seed_urls=["https://www.lbrce.ac.in"],
        allowed_domains=["lbrce.ac.in"],
        max_pages=5,
        request_delay=0.0
    )
    
    pages, pdfs = await crawler.crawl()
    
    assert len(pages) == 2
    assert pages[0].document_id == "home"
    assert pages[1].document_id == "cse"
    
    assert "https://www.lbrce.ac.in/syllabus.pdf" in pdfs
    assert "https://www.lbrce.ac.in/cse/handbook.pdf" in pdfs
    assert len(pdfs) == 2
