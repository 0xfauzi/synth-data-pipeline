import re
import json
import logging
from typing import List, Dict, Any
from datasketch import MinHash, MinHashLSH

logger = logging.getLogger(__name__)

class Deduplicator:
    """Remove near-duplicate examples using MinHash."""
    
    def __init__(self, threshold: float = 0.9, num_perm: int = 128):
        self.threshold = threshold
        self.num_perm = num_perm
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
    
    def _shingles(self, text: str, n: int = 5) -> set:
        """Create n-gram shingles from text."""
        tokens = re.findall(r"\w+", text.lower())
        return {" ".join(tokens[i:i+n]) for i in range(max(1, len(tokens)-n+1))}
    
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
