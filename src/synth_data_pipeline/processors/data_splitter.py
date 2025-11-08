import random
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataSplitter:
    """Create balanced train/val/test splits."""
    
    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
        seed: int = 42
    ):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify = stratify
        self.seed = seed
        random.seed(seed)
    
    def split(
        self,
        rows: List[Dict[str, Any]],
        labels: Optional[List[str]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split data into train/val/test sets."""
        
        random.shuffle(rows)
        
        if not self.stratify or labels is None:
            # Simple random split
            n = len(rows)
            n_test = int(n * self.test_ratio)
            n_val = int(n * self.val_ratio)
            
            test = rows[:n_test]
            val = rows[n_test:n_test + n_val]
            train = rows[n_test + n_val:]
            
        else:
            # Stratified split for multi-label
            train, val, test = [], [], []
            
            # Calculate per-label targets
            n = len(rows)
            n_test = int(n * self.test_ratio)
            n_val = int(n * self.val_ratio)
            
            val_counts = defaultdict(int)
            test_counts = defaultdict(int)
            
            # Calculate target counts per label
            label_targets_val = {}
            label_targets_test = {}
            
            for label in labels:
                positive_count = sum(
                    1 for r in rows
                    if "judgment" in r 
                    and "labels" in r["judgment"]
                    and r["judgment"]["labels"].get(label, 0) >= 0.5
                )
                label_targets_val[label] = max(1, int(positive_count * self.val_ratio))
                label_targets_test[label] = max(1, int(positive_count * self.test_ratio))
            
            # Assign rows to splits
            for row in rows:
                if "judgment" not in row or "labels" not in row["judgment"]:
                    train.append(row)
                    continue
                
                row_labels = row["judgment"]["labels"]
                positive_labels = [k for k, v in row_labels.items() if v >= 0.5]
                
                # Check if any positive labels need more examples in val/test
                needs_test = any(
                    test_counts[label] < label_targets_test[label]
                    for label in positive_labels
                )
                needs_val = any(
                    val_counts[label] < label_targets_val[label]
                    for label in positive_labels
                )
                
                if needs_test and len(test) < n_test:
                    test.append(row)
                    for label in positive_labels:
                        test_counts[label] += 1
                elif needs_val and len(val) < n_val:
                    val.append(row)
                    for label in positive_labels:
                        val_counts[label] += 1
                else:
                    train.append(row)
        
        logger.info(f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
        
        return train, val, test
