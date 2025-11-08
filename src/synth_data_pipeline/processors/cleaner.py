import re
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TextCleaner:
    """Clean and normalize text data."""

    DEFAULT_BOILERPLATE_PATTERNS = [
        r"^thanks for reading.*$",
        r"^#hiring\b.*$",
    ]

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("clean_text", True))
        self.unicode_normalize = bool(self.config.get("unicode_normalize", False))
        self.boilerplate_ruleset = self.config.get("boilerplate_ruleset")
        self._compiled_boilerplate = [
            re.compile(pattern, flags=re.IGNORECASE) for pattern in self.DEFAULT_BOILERPLATE_PATTERNS
        ]

        try:
            import unicodedata  # noqa: F401
            self._has_unicodedata = True
        except ImportError:  # pragma: no cover - should always exist
            self._has_unicodedata = False

    def _apply_unicode_normalization(self, text: str) -> str:
        if not self.unicode_normalize or not self._has_unicodedata:
            return text
        import unicodedata

        return unicodedata.normalize("NFKC", text)

    def _remove_boilerplate(self, text: str) -> str:
        if not self.boilerplate_ruleset:
            return text
        lines = []
        for line in text.splitlines():
            if any(pattern.match(line.strip()) for pattern in self._compiled_boilerplate):
                continue
            lines.append(line)
        return "\n".join(lines).strip()

    def clean_text(self, text: str) -> str:
        """Clean a single text string."""
        if not self.enabled or not isinstance(text, str):
            return text

        text = self._apply_unicode_normalization(text)

        # Remove excess whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Fix common encoding issues
        text = (
            text.replace("â€™", "'")
            .replace("â€œ", '"')
            .replace("â€", '"')
            .replace("Ã©", "é")
        )

        # Remove zero-width characters
        text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)

        # Normalize quotes
        text = re.sub(r'[""\'`´]', '"', text)

        text = self._remove_boilerplate(text)
        return text

    def clean_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Clean text fields in a row."""
        if "text" in row and isinstance(row["text"], str):
            row["text"] = self.clean_text(row["text"])

        if "generated" in row and isinstance(row["generated"], dict):
            if "text" in row["generated"]:
                row["generated"]["text"] = self.clean_text(row["generated"]["text"])
            if "opening_hook" in row["generated"]:
                row["generated"]["opening_hook"] = self.clean_text(row["generated"]["opening_hook"])

        return row

    def clean_batch(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean a batch of rows."""
        if not self.enabled:
            return rows
        return [self.clean_row(row) for row in rows]
