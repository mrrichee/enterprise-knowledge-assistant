"""
app/services/embedding_service.py
===================================
Phase 3: Embedding Generation

CONCEPT: What are embeddings?
Text embeddings are numerical vectors that capture semantic meaning.
"The cat sat on the mat" and "A feline rested on the rug" will have
very similar vectors even though they share no words.

HOW IT WORKS:
    "What is the refund policy?" → [0.12, -0.34, 0.87, ...] (384 numbers)

The model (all-MiniLM-L6-v2) maps any text to the same 384-dimensional
space, so we can measure how "close" two pieces of text are using
cosine similarity.

BATCHING:
We embed in batches of 32 for memory efficiency. Embedding 1000 chunks
one-by-one is 30x slower than batching them.
"""

import time
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import get_config
from app.core.models import Chunk
from app.core.exceptions import EmbeddingError
from app.utils.logger import get_logger
from app.utils.helpers import chunk_list, timer

logger = get_logger(__name__)
cfg    = get_config()

BATCH_SIZE = 32


class EmbeddingService:
    """
    Wraps sentence-transformers to convert text → numpy vectors.

    The model is loaded ONCE at construction time (it's ~90MB).
    Reloading it on every query would be extremely slow.
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL}")
        try:
            self._model = SentenceTransformer(cfg.EMBEDDING_MODEL)
            # Warm up the model with a dummy encode so the first real
            # request isn't slow
            self._model.encode(["warmup"], show_progress_bar=False)
            logger.info(f"✅ Embedding model loaded | dim={cfg.EMBEDDING_DIMENSION}")
        except Exception as e:
            raise EmbeddingError(f"Failed to load model '{cfg.EMBEDDING_MODEL}': {e}") from e

    @timer
    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        """
        Embed a list of Chunk objects into a 2D numpy array.

        Args:
            chunks: list of Chunk objects

        Returns:
            numpy array of shape (len(chunks), EMBEDDING_DIMENSION)
            Row i = embedding for chunks[i]

        CONCEPT: Why return numpy array and not a list?
        FAISS expects numpy float32 arrays. Returning the right type
        here avoids conversions later.
        """
        if not chunks:
            raise EmbeddingError("Cannot embed empty chunk list.")

        texts  = [chunk.content for chunk in chunks]
        logger.info(f"Embedding {len(texts)} chunks in batches of {BATCH_SIZE}")

        try:
            all_embeddings = []
            batches        = chunk_list(texts, BATCH_SIZE)

            for i, batch in enumerate(batches):
                embeddings = self._model.encode(
                    batch,
                    show_progress_bar = False,
                    convert_to_numpy  = True,
                    normalize_embeddings = True,
                    # normalize_embeddings=True makes cosine similarity
                    # equivalent to dot product — faster in FAISS
                )
                all_embeddings.append(embeddings)
                logger.debug(f"  Batch {i+1}/{len(batches)} embedded")

            result = np.vstack(all_embeddings).astype("float32")
            logger.info(f"✅ Embeddings ready: shape={result.shape}")
            return result

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(str(e)) from e

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string for retrieval.

        Returns shape (1, EMBEDDING_DIMENSION) — FAISS expects 2D input.
        normalize_embeddings=True must match what we used for chunks.
        """
        try:
            embedding = self._model.encode(
                [query],
                show_progress_bar    = False,
                convert_to_numpy     = True,
                normalize_embeddings = True,
            )
            return embedding.astype("float32")
        except Exception as e:
            raise EmbeddingError(f"Query embedding failed: {e}") from e
