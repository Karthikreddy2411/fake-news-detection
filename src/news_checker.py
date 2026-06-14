"""
src/news_checker.py
─────────────────────────────────────────────────────────────────────────────
External verification layer for the fake news detector.

Uses GNews (Google News scraper) to cross-reference user-submitted articles
against live news from reputable sources. Produces:
  - A list of matching articles (title, source, URL, date)
  - A cross-reference score / verdict
  - A combined trust indicator merging ML confidence + external evidence

No API key required.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ── Well-known reputable sources ──────────────────────────────────────────────
REPUTABLE_SOURCES = {
    "reuters", "reuters.com",
    "associated press", "apnews.com", "ap news",
    "bbc", "bbc.com", "bbc news",
    "the new york times", "nytimes.com",
    "the washington post", "washingtonpost.com",
    "the guardian", "theguardian.com",
    "cnn", "cnn.com",
    "npr", "npr.org",
    "al jazeera", "aljazeera.com",
    "abc news", "abcnews.go.com",
    "cbs news", "cbsnews.com",
    "nbc news", "nbcnews.com",
    "the wall street journal", "wsj.com",
    "usa today", "usatoday.com",
    "bloomberg", "bloomberg.com",
    "politico", "politico.com",
    "the hill", "thehill.com",
    "time", "time.com",
    "forbes", "forbes.com",
}


@dataclass
class MatchedArticle:
    """A single matching news article from Google News."""
    title: str
    source: str
    url: str
    published: Optional[str] = None
    is_reputable: bool = False


@dataclass
class VerificationResult:
    """Cross-reference verification result."""
    query: str
    articles: list[MatchedArticle] = field(default_factory=list)
    total_found: int = 0
    reputable_count: int = 0
    verdict: str = "Unknown"
    verdict_emoji: str = "❓"
    error: Optional[str] = None

    @property
    def cross_ref_score(self) -> float:
        """0.0 (no corroboration) to 1.0 (strongly corroborated)."""
        if self.total_found == 0:
            return 0.0
        # Weight reputable sources more heavily
        score = min(1.0, (self.reputable_count * 0.25) + (self.total_found * 0.05))
        return round(score, 2)


class NewsChecker:
    """
    Cross-references article text against Google News via GNews.

    Usage:
        checker = NewsChecker()
        result = checker.check("Biden signs new climate bill into law")
    """

    def __init__(self, max_results: int = 10, language: str = "en",
                 country: str = "US"):
        self.max_results = max_results
        self.language = language
        self.country = country

    def _extract_query(self, text: str) -> str:
        """
        Extract a search-friendly query from article text.
        Takes the first sentence (likely the lead / headline), cleans it,
        and truncates to a reasonable length for search.
        """
        # Strip extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Try to get the first 1-2 sentences (the lede)
        sentences = re.split(r"(?<=[.!?])\s+", text)
        lede = sentences[0] if sentences else text

        # Remove common clickbait / sensational words that pollute search
        noise_words = [
            r"\bBREAKING\b", r"\bEXCLUSIVE\b", r"\bURGENT\b",
            r"\bSHARE\b", r"\bWAKE UP\b", r"\bCENSORED\b",
            r"\bSECRETLY\b", r"\bMASSIVE\b", r"\bDESTROYS?\b",
            r"\bEXPOSED\b", r"\bSHOCKING\b",
        ]
        for pattern in noise_words:
            lede = re.sub(pattern, "", lede, flags=re.IGNORECASE)

        # Clean up
        lede = re.sub(r"[!]{2,}", "!", lede)
        lede = re.sub(r"\s+", " ", lede).strip()

        # Cap at ~120 chars for a good search query
        if len(lede) > 120:
            lede = lede[:120].rsplit(" ", 1)[0]

        return lede

    def _is_reputable(self, source_name: str, url: str = "") -> bool:
        """Check if a source is in the known reputable list."""
        name_lower = source_name.lower().strip()
        url_lower = url.lower()

        for rep in REPUTABLE_SOURCES:
            if rep in name_lower or rep in url_lower:
                return True
        return False

    def check(self, text: str) -> VerificationResult:
        """
        Cross-reference the given article text against Google News.

        Args:
            text: The article text (raw or cleaned).

        Returns:
            VerificationResult with matching articles and verdict.
        """
        query = self._extract_query(text)

        result = VerificationResult(query=query)

        try:
            from gnews import GNews

            gn = GNews(
                language=self.language,
                country=self.country,
                max_results=self.max_results,
            )

            raw_articles = gn.get_news(query)

            if not raw_articles:
                result.verdict = "No matching articles found"
                result.verdict_emoji = "⚠️"
                return result

            for art in raw_articles:
                source = art.get("publisher", {})
                source_name = source.get("title", "Unknown") if isinstance(source, dict) else str(source)
                source_url = source.get("href", "") if isinstance(source, dict) else ""

                url = art.get("url", "")
                is_rep = self._is_reputable(source_name, url or source_url)

                matched = MatchedArticle(
                    title=art.get("title", "Untitled"),
                    source=source_name,
                    url=url,
                    published=art.get("published date", art.get("published", None)),
                    is_reputable=is_rep,
                )
                result.articles.append(matched)

            result.total_found = len(result.articles)
            result.reputable_count = sum(1 for a in result.articles if a.is_reputable)

            # ── Determine verdict ─────────────────────────────────────────────
            if result.reputable_count >= 3:
                result.verdict = "Strongly corroborated by reputable sources"
                result.verdict_emoji = "✅"
            elif result.reputable_count >= 1:
                result.verdict = "Partially corroborated — found in some reputable sources"
                result.verdict_emoji = "🟡"
            elif result.total_found >= 3:
                result.verdict = "Found in multiple sources, but none are major outlets"
                result.verdict_emoji = "🟠"
            elif result.total_found >= 1:
                result.verdict = "Limited coverage — only found in a few sources"
                result.verdict_emoji = "🟠"
            else:
                result.verdict = "No matching reports found in any source"
                result.verdict_emoji = "🔴"

        except Exception as e:
            result.error = str(e)
            result.verdict = "Verification unavailable (network error)"
            result.verdict_emoji = "❌"

        return result

    def get_combined_assessment(self, ml_confidence: float, ml_label: str,
                                verification: VerificationResult) -> dict:
        """
        Combine ML prediction with external verification into a final assessment.

        Args:
            ml_confidence: Model's confidence (0.5-1.0)
            ml_label: "REAL" or "FAKE"
            verification: The VerificationResult from check()

        Returns:
            dict with combined_verdict, trust_level, explanation
        """
        cross_score = verification.cross_ref_score

        # ── Combine signals ───────────────────────────────────────────────────
        if ml_label == "REAL" and cross_score >= 0.5:
            trust_level = "HIGH"
            trust_color = "#10b981"
            explanation = (
                f"The model predicts this is REAL news ({ml_confidence:.0%} confidence) "
                f"and it's corroborated by {verification.reputable_count} reputable source(s)."
            )
        elif ml_label == "REAL" and cross_score < 0.5:
            trust_level = "MEDIUM"
            trust_color = "#f59e0b"
            explanation = (
                f"The model predicts REAL ({ml_confidence:.0%}), but limited external "
                f"corroboration was found. Exercise caution."
            )
        elif ml_label == "FAKE" and cross_score >= 0.5:
            trust_level = "MEDIUM"
            trust_color = "#f59e0b"
            explanation = (
                f"The model flags this as FAKE ({ml_confidence:.0%}), but similar stories "
                f"appear in reputable sources. The model may be wrong — verify manually."
            )
        elif ml_label == "FAKE" and cross_score < 0.3:
            trust_level = "HIGH"
            trust_color = "#ef4444"
            explanation = (
                f"The model predicts FAKE ({ml_confidence:.0%}) and no reputable sources "
                f"are reporting this story. Likely unreliable."
            )
        else:
            trust_level = "LOW"
            trust_color = "#f59e0b"
            explanation = (
                f"Mixed signals — model says {ml_label} ({ml_confidence:.0%}) with "
                f"inconclusive external verification."
            )

        return {
            "trust_level": trust_level,
            "trust_color": trust_color,
            "explanation": explanation,
            "cross_ref_score": cross_score,
            "ml_confidence": ml_confidence,
            "ml_label": ml_label,
        }


if __name__ == "__main__":
    checker = NewsChecker()
    result = checker.check(
        "The White House confirmed on Thursday that the administration will "
        "introduce a new policy framework aimed at reducing carbon emissions."
    )
    print(f"Query: {result.query}")
    print(f"Found: {result.total_found} articles ({result.reputable_count} reputable)")
    print(f"Verdict: {result.verdict_emoji} {result.verdict}")
    print(f"Score: {result.cross_ref_score}")
    for a in result.articles[:5]:
        tag = " ⭐" if a.is_reputable else ""
        print(f"  • [{a.source}] {a.title}{tag}")
