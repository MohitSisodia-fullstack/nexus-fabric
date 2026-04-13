# tests/test_chunking.py
import sys
sys.path.insert(0, "app")

import pytest
from notebooks.silver_chunk_logic import semantic_chunk, split_into_sentences, approximate_token_count


class TestApproximateTokenCount:
    def test_empty(self):
        assert approximate_token_count("") == 0

    def test_short_text(self):
        # 4 chars = ~1 token
        assert approximate_token_count("test") == 1

    def test_long_text(self):
        text = "word " * 200   # 1000 chars ≈ 250 tokens
        assert approximate_token_count(text) == 250


class TestSplitIntoSentences:
    def test_basic_split(self):
        text = "Hello world. This is a test. Another sentence!"
        sents = split_into_sentences(text)
        assert len(sents) >= 2

    def test_empty_text(self):
        assert split_into_sentences("") == []

    def test_single_sentence(self):
        text = "This is just one sentence"
        sents = split_into_sentences(text)
        assert len(sents) == 1
        assert sents[0] == text


class TestSemanticChunk:
    def test_short_text_single_chunk(self):
        text = "This is a short document. It has two sentences."
        chunks = semantic_chunk(text, chunk_size=512, overlap=64)
        assert len(chunks) == 1
        assert chunks[0]["chunk_index"] == 0
        assert "chunk_text" in chunks[0]

    def test_long_text_multiple_chunks(self):
        # Generate ~2000 tokens of text
        sentence = "The quick brown fox jumped over the lazy dog. " * 20
        text = sentence * 5
        chunks = semantic_chunk(text, chunk_size=100, overlap=20)
        assert len(chunks) > 1
        # Chunks should be indexed sequentially
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_overlap_carried(self):
        """Last sentences of chunk N should appear at start of chunk N+1."""
        sentence = "This is sentence number {i}. " 
        text = " ".join(f"This is sentence number {i}." for i in range(100))
        chunks = semantic_chunk(text, chunk_size=80, overlap=20)
        assert len(chunks) >= 2
        # Some overlap: words from end of chunk 0 should appear in chunk 1
        end_words   = set(chunks[0]["chunk_text"].split()[-10:])
        start_words = set(chunks[1]["chunk_text"].split()[:15])
        assert len(end_words & start_words) > 0, "Overlap not carried between chunks"

    def test_empty_text(self):
        assert semantic_chunk("") == []

    def test_chunk_ids_unique(self):
        text = " ".join(f"Sentence {i} with some content." for i in range(200))
        chunks = semantic_chunk(text, chunk_size=50, overlap=10)
        # chunk_index should be unique
        indices = [c["chunk_index"] for c in chunks]
        assert len(indices) == len(set(indices))


# tests/test_pii_guardrail.py
import sys
sys.path.insert(0, "app")

from guardrails import PIIGuardrail, scan_before_llm


class TestPIIGuardrail:
    def setup_method(self):
        self.g = PIIGuardrail()

    def test_clean_text_passes(self):
        result = self.g.scan("How does the medallion architecture work?")
        assert result.safe is True
        assert result.blocked is False
        assert result.pii_found == []

    def test_email_redacted(self):
        result = self.g.scan("Contact john.doe@example.com for more info.")
        assert result.safe is True
        assert result.blocked is False
        assert "EMAIL" in result.pii_found or "EMAIL_ADDRESS" in result.pii_found
        assert "john.doe@example.com" not in result.sanitized_text

    def test_ssn_blocks_request(self):
        result = self.g.scan("My SSN is 123-45-6789")
        assert result.blocked is True
        assert result.sanitized_text == "[BLOCKED]"

    def test_credit_card_blocks(self):
        result = self.g.scan("Card number 4111-1111-1111-1111")
        assert result.blocked is True

    def test_empty_text(self):
        result = self.g.scan("")
        assert result.safe is True

    def test_phone_redacted(self):
        result = self.g.scan("Call me at 555-867-5309 anytime.")
        # Should redact, not block
        assert result.blocked is False
        assert "555-867-5309" not in result.sanitized_text

    def test_ip_redacted(self):
        result = self.g.scan("Server IP is 192.168.1.100")
        assert result.blocked is False
        assert "192.168.1.100" not in result.sanitized_text
