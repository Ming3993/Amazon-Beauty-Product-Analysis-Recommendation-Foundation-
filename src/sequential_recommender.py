"""
Sequential Next-Item Recommender for E-Commerce / Beauty Care.
Models user journey progressions (Item_t -> Item_t+1) within a given time window (e.g. 1-90 days).
Enables next-routine step recommendation (e.g., Cleanser -> Serum, Straightener -> Heat Protection).
"""

from collections import Counter, defaultdict
import sqlite3
from typing import Any, Dict, List, Optional
import numpy as np


class SequentialRecommender:
    """Markov Chain / Transition-based Sequential Recommender."""

    def __init__(
        self,
        min_days: int = 1,
        max_days: int = 90,
        min_support: int = 2,
        db_path: str = "data/processed/products_metadata.db",
    ):
        self.min_days = min_days
        self.max_days = max_days
        self.min_support = min_support
        self.db_path = db_path
        self.transitions = defaultdict(Counter)  # asin_from -> {asin_to: count}
        self.total_outgoing = Counter()  # asin_from -> total count
        self.is_fitted = False

    def fit(self, dataset_path: str = "data/processed/dataset.npz"):
        """Fit transitions from interaction history."""
        data = np.load(dataset_path, allow_pickle=True)
        user_idx = data["user_idx"]
        prod_idx = data["prod_idx"]
        time_col = data["time_col"]
        prod_map = data["prod_map"]

        # Sort interactions chronologically per user
        sort_order = np.lexsort((time_col, user_idx))
        u_sorted = user_idx[sort_order]
        p_sorted = prod_idx[sort_order]
        t_sorted = time_col[sort_order].astype("datetime64[D]").astype(int)

        # Consecutive transitions within same user
        same_user = u_sorted[:-1] == u_sorted[1:]
        dt = t_sorted[1:] - t_sorted[:-1]
        valid = same_user & (dt >= self.min_days) & (dt <= self.max_days)

        from_items = p_sorted[:-1][valid]
        to_items = p_sorted[1:][valid]

        # Populate transition counts
        for f_idx, t_idx in zip(from_items, to_items):
            if f_idx != t_idx:  # Cross-item recommendation
                asin_from = prod_map[f_idx]
                asin_to = prod_map[t_idx]
                self.transitions[asin_from][asin_to] += 1
                self.total_outgoing[asin_from] += 1

        self.is_fitted = True
        return self

    def _get_metadata(self, asin: str) -> Dict[str, Any]:
        """Fetch item metadata from SQLite database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                row = c.execute(
                    "SELECT title, brand, price, category FROM products WHERE asin = ?",
                    (asin,),
                ).fetchone()
                if row:
                    return {
                        "asin": asin,
                        "title": row[0] or "N/A",
                        "brand": row[1] or "N/A",
                        "price": row[2] or 0.0,
                        "category": row[3] or "N/A",
                    }
        except Exception:
            pass
        return {
            "asin": asin,
            "title": "N/A",
            "brand": "N/A",
            "price": 0.0,
            "category": "N/A",
        }

    def recommend_next(
        self, asin: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Predict the next items a customer will likely buy after purchasing `asin`."""
        if asin not in self.transitions:
            return []

        outcomes = self.transitions[asin]
        total = self.total_outgoing[asin]

        # Filter by minimum support and sort by count descending
        sorted_next = [
            (target, count)
            for target, count in outcomes.most_common()
            if count >= self.min_support
        ][:top_k]

        results = []
        for target_asin, count in sorted_next:
            prob = count / total if total > 0 else 0.0
            meta = self._get_metadata(target_asin)
            meta["transition_count"] = count
            meta["probability"] = round(prob, 4)
            results.append(meta)

        return results

    def recommend_for_sequence(
        self, sequence: List[str], top_k: int = 5, decay: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Given a user's recent purchase sequence [item_1, item_2, ..., item_last],

        predict the most probable next items with exponential recency decay.
        """
        scores = defaultdict(float)
        # Weight more recent items higher: weight = decay ** (len(sequence) - 1 - i)
        for i, asin in enumerate(sequence):
            weight = decay ** (len(sequence) - 1 - i)
            if asin in self.transitions:
                total = self.total_outgoing[asin]
                for target_asin, count in self.transitions[asin].items():
                    if target_asin not in sequence:  # Don't recommend items already bought
                        scores[target_asin] += weight * (count / total)

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        results = []
        for target_asin, score in sorted_items:
            meta = self._get_metadata(target_asin)
            meta["sequence_score"] = round(score, 4)
            results.append(meta)

        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sequential Next-Item Recommender")
    parser.add_argument("--asin", type=str, default=None, help="Source ASIN to recommend next items for")
    parser.add_argument("--top-k", type=int, default=5, help="Number of items to recommend")
    args = parser.parse_args()

    recommender = SequentialRecommender()
    print("Fitting Sequential Next-Item Recommender...")
    recommender.fit()
    print(f"Total source items with transitions: {len(recommender.transitions):,}")

    test_asins = [args.asin] if args.asin else [
        "B001MA0QY2",  # HSI Flat Iron Straightener
        "B007SZCG0C",  # Ultrax Labs Hair Loss Shampoo
        "B00IBS9QC6",  # Dead Sea Mud Facial Mask
    ]

    for asin in test_asins:
        source_meta = recommender._get_metadata(asin)
        print(f"\n=======================================================")
        print(f"Source Item: [{source_meta['brand']}] {source_meta['title'][:55]}")
        print(f"ASIN: {asin} | Price: ${source_meta['price']:.2f} | Category: {source_meta['category']}")
        print(f"------------------- Top Next Items ------------------")
        next_items = recommender.recommend_next(asin, top_k=args.top_k)
        if not next_items:
            print("  (No sequential transitions above support threshold)")
        for rank, item in enumerate(next_items, 1):
            print(f" {rank}. [{item['brand']}] {item['title'][:45]}")
            print(f"    P(Next|Current) = {item['probability']*100:.1f}% ({item['transition_count']} times) | Price: ${item['price']:.2f}")

