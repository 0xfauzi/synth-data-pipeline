import re
import json
import logging
from typing import List, Dict, Any
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

class Deduplicator:
    """Remove near-duplicate examples using MinHash."""
    
    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        n_gram: int = 5,
        shingle_level: str = "word",
        method: str = "minhash_lsh",
    ):
        self.threshold = threshold
        self.num_perm = num_perm
        self.n_gram = max(1, int(n_gram))
        self.shingle_level = shingle_level
        self.method = method
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.stats = {"total": 0, "unique": 0, "duplicates": 0}
    
    def _get_text(self, row: Dict[str, Any]) -> str:
        """Extract text from row."""
        if "text" in row:
            return row["text"]
        elif "generated" in row and "text" in row["generated"]:
            return row["generated"]["text"]
        else:
            return json.dumps(row)
    
    def _shingles(self, text: str) -> set:
        """Create n-gram shingles from text."""
        if self.shingle_level == "char":
            cleaned = re.sub(r"\s+", " ", text.lower())
            return {
                cleaned[i : i + self.n_gram]
                for i in range(max(1, len(cleaned) - self.n_gram + 1))
            }
        tokens = re.findall(r"\w+", text.lower())
        return {" ".join(tokens[i : i + self.n_gram]) for i in range(max(1, len(tokens) - self.n_gram + 1))}
    
    def _minhash(self, text: str) -> MinHash:
        """Create MinHash from text."""
        m = MinHash(num_perm=self.num_perm)
        for shingle in self._shingles(text):
            m.update(shingle.encode("utf8"))
        return m
    
    def deduplicate(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove near-duplicates from rows."""
        unique_rows = []
        
        for i, row in enumerate(rows):
            self.stats["total"] += 1
            text = self._get_text(row)
            if not text:
                logger.debug("Deduplicator received row with empty text; preserving without dedup.")
                unique_rows.append(row)
                self.stats["unique"] += 1
                continue
            mhash = self._minhash(text)

            # Check if near-duplicate exists
            if self.lsh.query(mhash):
                self.stats["duplicates"] += 1
                continue

            # Add to index
            key = f"row-{i}"
            self.lsh.insert(key, mhash)
            unique_rows.append(row)
            self.stats["unique"] += 1
        
        logger.info(f"Deduplication stats: {self.stats}")
        return unique_rows
