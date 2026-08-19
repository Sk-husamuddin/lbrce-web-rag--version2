"""
Image Parser using a vision-capable LLM (via OpenRouter) for LBRCE timetable
and other image-based document extraction.

Mirrors PDFParser's interface (ImageDocument ~ PDFDocument) so it plugs into
the existing chunk_document() / ingestion pipeline with no changes required
there. Prototype validated against a real LBRCE timetable image on
google/gemma-4-31b-it:free -- see prototype_timetable_ocr.py.

Two API keys are rotated round-robin across calls to roughly double the
effective free-tier rate limit when processing many images in one batch.
"""

import base64
import itertools
import logging
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from openai import OpenAI
from PIL import Image

from backend.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class ImageDocument:
    """Represents a parsed image document. Same shape as PDFDocument."""
    document_id: str
    title: str
    content: str
    source_url: str
    source_type: str = "image"
    department: Optional[str] = None
    metadata: dict = field(default_factory=dict)


_SYSTEM_PROMPT = """You are an expert at reading Indian college timetable images \
and converting them into clean, structured plain text for a search index. \
You will be shown one timetable image. Follow these rules exactly:

1. First extract header metadata: college name, department, course, \
semester, section, academic year, classroom, regulation, effective date.

2. Then extract the WEEKLY SCHEDULE. The table uses merged cells with \
left/right arrows (e.g. "<- CN LAB ->") to show one activity spanning \
multiple time periods. You MUST expand every merged cell into one line \
per individual time slot it covers -- do not leave any slot's activity \
ambiguous or grouped. Use this exact format per day:

Monday:
09:00-10:00 -- CN LAB
10:00-11:00 -- CN LAB
...

Cover all six days shown (Monday through Saturday), all time periods \
shown in the grid, and mark the lunch period explicitly.

3. Then extract the COURSE INFORMATION table (course code, short code, \
full course name, instructor name(s)) as a plain list, one course per \
block.

4. Ignore handwritten signatures, stamps, and decorative borders -- \
do not transcribe or describe them.

5. Output ONLY the structured text described above. No preamble, no \
markdown code fences, no commentary about the image itself.
"""

_USER_PROMPT = (
    "Convert this timetable image into the structured text format described "
    "in your instructions."
)


class ImageParser:
    """
    Parses timetable/document images using a vision LLM via OpenRouter.

    Rotates between up to two API keys (OPENROUTER_API_KEY, OPENROUTER_API_KEY_2)
    round-robin, per call, to spread load across two accounts' free-tier
    rate limits during a large batch ingestion run.
    """

    MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB raw download cap
    VISION_MODEL = "google/gemma-4-31b-it:free"
    PROMPT_VERSION = "timetable-v1"

    def __init__(
        self,
        base_url: Optional[str] = None,
        max_dimension: int = 1600,
        jpeg_quality: int = 85,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or settings.OPENROUTER_BASE_URL).rstrip("/")
        self.max_dimension = max_dimension
        self.jpeg_quality = jpeg_quality
        self.max_retries = max_retries

        keys = [settings.OPENROUTER_API_KEY, settings.OPENROUTER_API_KEY_2]
        self._api_keys = [k for k in keys if k]
        if not self._api_keys:
            raise ValueError("No OPENROUTER_API_KEY(_2) found in settings/.env.")
        self._key_cycle = itertools.cycle(self._api_keys)

    def _next_client(self) -> OpenAI:
        """Round-robin to the next API key for this call."""
        return OpenAI(base_url=self.base_url, api_key=next(self._key_cycle))

    def _encode_image(self, image_data: bytes) -> str:
        """Resize (if needed) and return a base64 data URL for the image."""
        with Image.open(BytesIO(image_data)) as img:
            img = img.convert("RGB")
            width, height = img.size
            if max(width, height) > self.max_dimension:
                scale = self.max_dimension / max(width, height)
                img = img.resize(
                    (int(width * scale), int(height * scale)), Image.LANCZOS
                )
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=self.jpeg_quality)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"

    def parse(self, image_data: bytes, url: str) -> Optional[ImageDocument]:
        """
        Parse image bytes via the vision LLM and return an ImageDocument.

        Args:
            image_data: Raw image file bytes.
            url: Source URL for metadata.

        Returns:
            ImageDocument with extracted content, or None if extraction failed
            after all retries.
        """
        if len(image_data) > self.MAX_IMAGE_SIZE:
            logger.warning(f"Image too large ({len(image_data)} bytes): {url}")
            return None

        try:
            data_url = self._encode_image(image_data)
        except Exception as e:
            logger.error(f"Failed to open/resize image from {url}: {e}")
            return None

        text = self._call_vision_model(data_url, url)
        if not text:
            return None

        return self.document_from_text(text, url, extraction_method="vision_llm")

    @classmethod
    def document_from_text(
        cls,
        text: str,
        url: str,
        extraction_method: str = "cached_vision_llm",
    ) -> ImageDocument:
        """Build an ImageDocument without making another vision-model call."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        document_id = re.sub(r"[^a-zA-Z0-9/_-]", "_", path) if path else "image"
        title = parsed.path.rstrip("/").split("/")[-1] or "Untitled Image"
        return ImageDocument(
            document_id=document_id.replace("/", "_")[:100],
            title=title,
            content=text,
            source_url=url,
            source_type="image",
            department=cls._department_from_url(url),
            metadata={
                "extraction_method": extraction_method,
                "model": cls.VISION_MODEL,
                "prompt_version": cls.PROMPT_VERSION,
                "content_length": len(text),
            },
        )

    def _call_vision_model(self, data_url: str, url: str) -> Optional[str]:
        """Call the vision LLM with retry-on-transient-error, key rotation per attempt."""
        for attempt in range(1, self.max_retries + 1):
            client = self._next_client()
            try:
                completion = client.chat.completions.create(
                    model=self.VISION_MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": _USER_PROMPT},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        },
                    ],
                    max_tokens=2000,
                )
            except Exception as e:
                logger.warning(
                    f"Vision call raised (attempt {attempt}/{self.max_retries}) for {url}: {e}"
                )
                completion = None

            choices = getattr(completion, "choices", None) if completion else None
            if choices:
                return (choices[0].message.content or "").strip()

            error = getattr(completion, "error", None) if completion else None
            code = error.get("code") if isinstance(error, dict) else None
            if code in (429, 504) and attempt < self.max_retries:
                wait_s = 8 * attempt
                logger.info(
                    f"Transient error ({code}) for {url} — retrying in {wait_s}s with next key"
                )
                time.sleep(wait_s)
                continue

            logger.error(f"Vision extraction failed for {url}: {error or 'unknown error'}")
            return None

        return None

    @staticmethod
    def _generate_document_id(url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        doc_id = re.sub(r"[^a-zA-Z0-9/_-]", "_", path) if path else "image"
        return doc_id.replace("/", "_")[:100]

    @staticmethod
    def _infer_title(url: str) -> str:
        filename = urlparse(url).path.rstrip("/").split("/")[-1]
        return filename or "Untitled Image"

    @staticmethod
    def _department_from_url(url: str) -> Optional[str]:
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        dept_keywords = {
            "cse": "Computer Science Engineering", "cs": "Computer Science Engineering",
            "it": "Information Technology", "ece": "Electronics and Communication Engineering",
            "eee": "Electrical and Electronics Engineering", "mech": "Mechanical Engineering",
            "civil": "Civil Engineering", "ai": "Artificial Intelligence", "ml": "Machine Learning",
            "ds": "Data Science", "mba": "Master of Business Administration",
            "mca": "Master of Computer Applications", "h&s": "Humanities and Sciences",
        }
        for part in path_parts:
            if part.lower() in dept_keywords:
                return dept_keywords[part.lower()]
        return None

    def _infer_department(self, url: str) -> Optional[str]:
        return self._department_from_url(url)


def parse_image_from_url(url: str, timeout: float = 30.0) -> Optional[ImageDocument]:

    """Convenience function to fetch and parse an image from a URL."""
    import httpx

    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("image/"):
            logger.warning(f"Non-image content type: {content_type} for {url}")
            return None
        parser = ImageParser()
        return parser.parse(response.content, url)
    except Exception as e:
        logger.error(f"Failed to fetch/parse image from {url}: {e}")
        return None