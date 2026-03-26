"""
文本分 chunk

策略：
- 按段落分割，合并短段落，截断长段落
- 每个 chunk 目标 200-500 tokens
- chunk 之间有 50 token 重叠，避免信息截断
- 保留来源信息（source_id, page）
"""

import re
import hashlib
from typing import List

import tiktoken

from agent.schemas import TextChunk, DocumentMeta


class _WhitespaceTokenizer:
    def encode(self, text: str):
        return re.findall(r"\S+", text)

    def decode(self, tokens):
        return " ".join(tokens)


class TextChunker:
    def __init__(
        self,
        target_chunk_size: int = 400,   # tokens
        max_chunk_size: int = 600,
        overlap_size: int = 50,
        encoding_name: str = "cl100k_base",
    ):
        self.target_size = target_chunk_size
        self.max_size = max_chunk_size
        self.overlap_size = overlap_size
        try:
            self.tokenizer = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.tokenizer = _WhitespaceTokenizer()

    def chunk_document(self, meta: DocumentMeta, text: str) -> List[TextChunk]:
        """将一篇文档切成 chunks"""
        paragraphs = self._split_paragraphs(text)
        merged = self._merge_short_paragraphs(paragraphs)
        chunks = self._apply_overlap(merged)

        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = self._generate_chunk_id(meta.source_id, i)
            token_count = len(self.tokenizer.encode(chunk_text))
            result.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source_id=meta.source_id,
                page=self._extract_page_number(chunk_text),
                section=None,
                token_count=token_count,
            ))

        return result

    def _split_paragraphs(self, text: str) -> List[str]:
        """按双换行分割段落"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _merge_short_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """将短段落合并到目标大小"""
        merged = []
        buffer = ""

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            # 如果单段落就超过 max_size，需要强制截断
            if para_tokens > self.max_size:
                if buffer:
                    merged.append(buffer.strip())
                    buffer = ""
                split_parts = self._force_split(para)
                merged.extend(split_parts)
                continue

            buffer_tokens = len(self.tokenizer.encode(buffer)) if buffer else 0

            if buffer_tokens + para_tokens <= self.target_size:
                buffer = buffer + "\n\n" + para if buffer else para
            else:
                if buffer:
                    merged.append(buffer.strip())
                buffer = para

        if buffer:
            merged.append(buffer.strip())

        return merged

    def _force_split(self, text: str) -> List[str]:
        """对超长段落按句子边界截断"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        buffer = ""

        for sent in sentences:
            buffer_tokens = len(self.tokenizer.encode(buffer)) if buffer else 0
            sent_tokens = len(self.tokenizer.encode(sent))

            if buffer_tokens + sent_tokens <= self.target_size:
                buffer = buffer + " " + sent if buffer else sent
            else:
                if buffer:
                    chunks.append(buffer.strip())
                buffer = sent

        if buffer:
            chunks.append(buffer.strip())

        return chunks

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """在相邻 chunk 之间添加重叠"""
        if len(chunks) <= 1:
            return chunks

        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tokens = self.tokenizer.encode(chunks[i - 1])
            overlap_text = self.tokenizer.decode(prev_tokens[-self.overlap_size:])
            overlapped.append(overlap_text + " " + chunks[i])

        return overlapped

    def _generate_chunk_id(self, source_id: str, index: int) -> str:
        raw = f"{source_id}_{index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _extract_page_number(self, text: str) -> int | None:
        match = re.search(r'\[Page (\d+)\]', text)
        return int(match.group(1)) if match else None
