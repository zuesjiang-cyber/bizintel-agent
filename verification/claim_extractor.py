"""
从 memo 文本中提取可验证的 claim

每条 claim 包含：
- 原文文本
- 引用的来源 ID
- 是否包含数字
- 提取出的数字
"""

import re
from typing import List
import hashlib

from agent.schemas import Claim


class ClaimExtractor:
    """
    从分析文本中提取 claims

    策略：
    1. 按句子分割
    2. 过滤掉纯观点/连接句（太短的、以"However"/"Additionally"开头的连接句）
    3. 提取引用标记 [Source: xxx]
    4. 检测并提取数字
    """

    # 数字模式
    NUMBER_PATTERN = re.compile(
        r'\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|[BMK%]|bn|mn)?',
        re.IGNORECASE
    )

    # 引用模式
    CITATION_PATTERN = re.compile(r'\[Source:\s*([^\]]+)\]')

    def extract_claims(self, text: str, section_name: str) -> List[Claim]:
        sentences = self._split_sentences(text)
        claims = []

        for sent in sentences:
            if self._is_filler(sent):
                continue

            cited_sources = self.CITATION_PATTERN.findall(sent)
            numbers = self.NUMBER_PATTERN.findall(sent)

            # 清理句子中的引用标记（留纯文本）
            clean_text = self.CITATION_PATTERN.sub('', sent).strip()

            if len(clean_text) < 15:
                continue

            claim_id = hashlib.md5(clean_text.encode()).hexdigest()[:10]

            claims.append(Claim(
                claim_id=claim_id,
                text=clean_text,
                section=section_name,
                cited_sources=[s.strip() for s in cited_sources],
                contains_numbers=len(numbers) > 0,
                extracted_numbers=numbers,
            ))

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        # 按句号、问号、感叹号分割，但不在缩写处分割
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_filler(self, sentence: str) -> bool:
        """过滤掉非事实性的连接句"""
        filler_starts = [
            "in summary", "overall", "to summarize", "in conclusion",
            "it is worth noting", "it should be noted",
        ]
        lower = sentence.lower().strip()

        if len(lower) < 20:
            return True

        for filler in filler_starts:
            if lower.startswith(filler):
                return True

        return False
