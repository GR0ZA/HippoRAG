from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig

logger = get_logger(__name__)


class Qwen3EmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self,
        global_config=None,
        embedding_model_name: Optional[str] = None,
    ) -> None:
        super().__init__(global_config=global_config)

        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name

        config_dict = {
            "batch_size": self.global_config.embedding_batch_size,
            "normalize": self.global_config.embedding_return_as_normalized,
            "model_name": self.embedding_model_name,
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict)

        logger.debug(f"Loading SBert model '{self.embedding_model_name}'")
        self.model = SentenceTransformer(self.embedding_model_name)

        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.debug(f"SBert embedding dim = {self.embedding_dim}")

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """
        Encodes a list of texts into embeddings.
        Returns an (N x D) numpy array.
        """
        if isinstance(texts, str):
            texts = [texts]

        batch_size = self.embedding_config.batch_size
        normalize = self.embedding_config.normalize

        logger.debug(f"SBertEmbeddingModel.batch_encode: {len(texts)} texts, batch_size={batch_size}")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
            embeddings = embeddings / norms

        return embeddings
