"""
app/services/vector_store.py
=============================
Phase 3: FAISS Indexing & Semantic Retrieval

CONCEPT: What is FAISS?
Facebook AI Similarity Search (FAISS) is a library for efficient
nearest-neighbour search in high-dimensional vector spaces.

Given a query vector, FAISS finds the K most similar vectors
(highest cosine similarity) from millions of stored vectors — in milliseconds.

OUR SETUP:
- Index type: IndexFlatIP (Inner Product on normalised vectors = cosine similarity)
- We use IndexFlatIP instead of IndexFlatL2 because we normalized embeddings,
  making inner product = cosine similarity
- We also maintain a metadata store (dict) mapping FAISS integer IDs → Chunk objects
  because FAISS only stores vectors, not the text they came from

PERSISTENCE:
FAISS indexes can be saved to disk and reloaded — the user's documents
persist between app restarts. We save after every add operation.
"""

import pickle
import threading
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from app.core.config import get_config
from app.core.models import Chunk, Citation
from app.core.exceptions import VectorStoreError, NoRelevantChunksError
from app.utils.logger import get_logger
from app.utils.helpers import timer

logger = get_logger(__name__)
cfg    = get_config()

INDEX_FILE    = cfg.FAISS_DIR / "index.faiss"
METADATA_FILE = cfg.FAISS_DIR / "metadata.pkl"


class VectorStore:
    """
    Manages the FAISS index and the chunk metadata store.

    TWO parallel data structures:
        _index:    FAISS index — maps integer ID → 384-dim float32 vector
        _metadata: Python dict — maps integer ID → Chunk object

    They stay in sync: when we add chunk i, its embedding goes into
    _index at position i and its Chunk object goes into _metadata[i].
    """

    def __init__(self):
        cfg.FAISS_DIR.mkdir(parents=True, exist_ok=True)
        self._index    : faiss.Index        = None
        self._metadata : dict[int, Chunk]   = {}
        self._lock     = threading.RLock()   # Streamlit sessions run in threads
        self._load_or_create()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _load_or_create(self) -> None:
        """Load existing index from disk, or create a fresh one."""
        if INDEX_FILE.exists() and METADATA_FILE.exists():
            try:
                self._index    = faiss.read_index(str(INDEX_FILE))
                with open(METADATA_FILE, "rb") as f:
                    self._metadata = pickle.load(f)
                if self._index.ntotal != len(self._metadata):
                    raise ValueError(
                        f"index/metadata out of sync "
                        f"({self._index.ntotal} vectors vs {len(self._metadata)} chunks)"
                    )
                logger.info(
                    f"✅ Loaded FAISS index: {self._index.ntotal} vectors, "
                    f"{len(self._metadata)} chunks"
                )
                return
            except Exception as e:
                logger.warning(f"Failed to load existing index ({e}), creating fresh one.")

        self._create_new_index()

    def _create_new_index(self) -> None:
        """Create a new empty FAISS index."""
        # IndexFlatIP = exact nearest-neighbour using Inner Product
        # "Flat" means brute-force (no approximation) — perfect for < 100k vectors
        self._index    = faiss.IndexFlatIP(cfg.EMBEDDING_DIMENSION)
        self._metadata = {}
        logger.info(f"Created new FAISS index (dim={cfg.EMBEDDING_DIMENSION})")

    # ── Add Documents ─────────────────────────────────────────────────────────

    @timer
    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the index.

        Args:
            chunks:     list of Chunk objects (length N)
            embeddings: numpy array of shape (N, 384)
        """
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                "add_chunks",
                f"Mismatch: {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        try:
            with self._lock:
                start_id = self._index.ntotal   # current size = first new ID

                # Add embeddings to FAISS (assigns sequential integer IDs)
                self._index.add(np.ascontiguousarray(embeddings, dtype="float32"))

                # Store chunk metadata with matching IDs
                for i, chunk in enumerate(chunks):
                    self._metadata[start_id + i] = chunk

                self._persist()
                logger.info(
                    f"✅ Added {len(chunks)} chunks to FAISS | "
                    f"Total: {self._index.ntotal}"
                )
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError("add_chunks", str(e)) from e

    # ── Search ────────────────────────────────────────────────────────────────

    def search_with_scores(
        self,
        query_embedding : np.ndarray,
        top_k           : int = None,
        doc_ids_filter  : Optional[list[str]] = None,
        min_fallback    : int = 0,
    ) -> list[tuple[Chunk, float]]:
        """
        Core retrieval. Returns [(Chunk, score)] sorted best-first.

        Chunks scoring below SIMILARITY_THRESHOLD are dropped, but if that
        leaves fewer than `min_fallback` results, the best-scoring chunks are
        returned anyway so the LLM can still look at the closest text
        (vague questions like "tell me the eligibility" often score low).
        """
        with self._lock:
            if self._index.ntotal == 0:
                return []

            top_k = top_k or cfg.RETRIEVAL_TOP_K
            # With a filter, scan the whole (small) index so excluded docs
            # can never crowd out valid results.
            search_k = self._index.ntotal if doc_ids_filter is not None else min(top_k, self._index.ntotal)

            try:
                query_embedding = np.ascontiguousarray(query_embedding, dtype="float32")
                scores, indices = self._index.search(query_embedding, search_k)

                allowed = set(doc_ids_filter) if doc_ids_filter is not None else None
                ranked: list[tuple[Chunk, float]] = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx == -1:
                        continue
                    chunk = self._metadata.get(int(idx))
                    if chunk is None:
                        continue
                    if allowed is not None and chunk.doc_id not in allowed:
                        continue
                    ranked.append((chunk, float(score)))

                results = [r for r in ranked if r[1] >= cfg.SIMILARITY_THRESHOLD][:top_k]
                if len(results) < min_fallback:
                    results = ranked[:min_fallback]

                logger.info(
                    f"Search returned {len(results)} chunks "
                    f"(threshold={cfg.SIMILARITY_THRESHOLD})"
                )
                return results
            except Exception as e:
                raise VectorStoreError("search", str(e)) from e

    @staticmethod
    def to_citation(chunk: Chunk, score: float) -> Citation:
        return Citation(
            filename    = chunk.filename,
            page_number = chunk.page_number,
            chunk_index = chunk.chunk_index,
            relevance   = max(0.0, score),
            excerpt     = chunk.content[:300],
        )

    def search(self, query_embedding, top_k=None, doc_ids_filter=None) -> list[Citation]:
        """Like search_with_scores() but returns Citation objects."""
        return [
            self.to_citation(c, s)
            for c, s in self.search_with_scores(query_embedding, top_k, doc_ids_filter)
        ]

    def search_chunks(self, query_embedding, top_k=None, doc_ids_filter=None) -> list[Chunk]:
        """Like search_with_scores() but returns only Chunk objects."""
        return [c for c, _ in self.search_with_scores(query_embedding, top_k, doc_ids_filter)]

    # ── Remove Document ───────────────────────────────────────────────────────

    def remove_document(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to a document.

        IndexFlatIP can't delete in place, so we rebuild the index from the
        stored vectors of the chunks we keep (reconstruct_n works on Flat
        indexes). Vectors and metadata stay perfectly aligned.

        Returns:
            Number of chunks removed
        """
        with self._lock:
            keep_ids = [i for i, c in sorted(self._metadata.items()) if c.doc_id != doc_id]
            removed_count = len(self._metadata) - len(keep_ids)

            if removed_count == 0:
                logger.warning(f"No chunks found for doc_id={doc_id}")
                return 0

            try:
                new_index    = faiss.IndexFlatIP(cfg.EMBEDDING_DIMENSION)
                new_metadata = {}
                if keep_ids:
                    all_vecs = self._index.reconstruct_n(0, self._index.ntotal)
                    new_index.add(np.ascontiguousarray(all_vecs[keep_ids], dtype="float32"))
                    new_metadata = {new_i: self._metadata[old_i] for new_i, old_i in enumerate(keep_ids)}

                self._index, self._metadata = new_index, new_metadata
                self._persist()
            except Exception as e:
                raise VectorStoreError("remove_document", str(e)) from e

            logger.info(f"Removed {removed_count} chunks for doc_id={doc_id}")
            return removed_count

    def doc_ids(self) -> set[str]:
        """IDs of all documents that currently have chunks in the index."""
        with self._lock:
            return {c.doc_id for c in self._metadata.values()}

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return index statistics for the analytics dashboard."""
        doc_ids      = {c.doc_id for c in self._metadata.values()}
        return {
            "total_vectors" : self._index.ntotal,
            "total_chunks"  : len(self._metadata),
            "total_documents": len(doc_ids),
            "index_type"    : type(self._index).__name__,
            "dimension"     : cfg.EMBEDDING_DIMENSION,
        }

    def is_empty(self) -> bool:
        return self._index.ntotal == 0

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self) -> None:
        """Save index and metadata to disk."""
        try:
            faiss.write_index(self._index, str(INDEX_FILE))
            with open(METADATA_FILE, "wb") as f:
                pickle.dump(self._metadata, f)
        except Exception as e:
            logger.error(f"Failed to persist FAISS index: {e}")
            raise VectorStoreError("persist", str(e)) from e
