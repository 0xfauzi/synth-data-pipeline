import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TextCleaner:
    """Clean and normalize text data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common encoding issues
        text = text.replace('â€™', "'").replace('â€œ', '"').replace('â€', '"')
        
        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`´]', '"', text)
        
        return text
    
    def clean_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Clean text fields in a row."""
        if "text" in row:
            row["text"] = self.clean_text(row["text"])
        
        if "generated" in row and "text" in row["generated"]:
            row["generated"]["text"] = self.clean_text(row["generated"]["text"])
        
        return row
    
    def clean_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean a batch of rows."""
        return [self.clean_row(row) for row in rows]
