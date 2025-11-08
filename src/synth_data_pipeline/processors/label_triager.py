import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LabelTriager:
    """Detect potential label quality issues."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.stats = {"total": 0, "flagged": 0}
    
    def find_issues(self, rows: List[Dict[str, Any]], labels: Optional[List[str]] = None) -> List[int]:
        """Find rows with potential label issues."""
        try:
            from cleanlab.multilabel_classification import find_label_issues
        except ImportError:
            logger.warning("Cleanlab not installed, skipping label triage")
            return []
        
        if not rows:
            return []
        
        # Extract labels and probabilities
        if labels is None:
            # Auto-detect labels from first row
            if "judgment" in rows[0] and "labels" in rows[0]["judgment"]:
                labels = list(rows[0]["judgment"]["labels"].keys())
            else:
                logger.warning("Could not detect label structure")
                return []
        
        Y, P = [], []
        for row in rows:
            if "judgment" in row and "labels" in row["judgment"]:
                probs = [row["judgment"]["labels"].get(label, 0.0) for label in labels]
                hard = [1 if p >= self.threshold else 0 for p in probs]
                P.append(probs)
                Y.append(hard)
        
        if not Y:
            return []
        
        Y = np.array(Y)
        P = np.array(P)
        
        # Find issues
        issues = find_label_issues(labels=Y, pred_probs=P)
        flagged = np.where(issues["is_label_issue"])[0].tolist()
        
        self.stats["total"] = len(rows)
        self.stats["flagged"] = len(flagged)
        
        logger.info(f"Label triage stats: {self.stats}")
        return flagged
    
    def filter_clean(self, rows: List[Dict[str, Any]], labels: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Return only rows without label issues."""
        issues = self.find_issues(rows, labels)
        issue_set = set(issues)
        return [row for i, row in enumerate(rows) if i not in issue_set]
