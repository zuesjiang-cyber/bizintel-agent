"""
从 memo 文本中提取可验证的 claim

每条 claim 包含：
- 原文文本
- 引用的来源 ID
- 是否包含数字
- 提取出的数字
- 特异性得分 (Specificity Score)
"""

import re
from typing import List
import hashlib

from agent.schemas import Claim


class ClaimExtractor:
    """
    从分析文本中提取 claims 并进行过滤打分
    
    1. 按句子分割
    2. 过滤主观/过渡/假设/信息不足等废话
    3. 提取引用 [Source: xxx]，若无则标为 no_citation
    4. 提取数字并计算 specificity_score
    5. 只保留 specificity_score > 0.3 的 claim
    """

    # 数字模式（支持 $ 符号、逗号、百分比、billion/million 等）
    NUMBER_PATTERN = re.compile(
        r'\$?[\d,]+\.?\d*\s*(?:billion|million|thousand|bn|mn|b|m|k|%)?',
        re.IGNORECASE
    )

    # 日期/年份模式
    DATE_PATTERN = re.compile(
        r'\b(?:19|20)\d{2}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:st|nd|rd|th)?,? \d{4}\b|\bQ[1-4]\s*\d{4}\b',
        re.IGNORECASE
    )

    # 引用模式
    CITATION_PATTERN = re.compile(r'\[Source:\s*([^\]]+)\]')

    # 专有名词简易启发式（大写字母开头的词，排除句首词）
    # 在这个 MVP 中，我们通过检测句子里除第一个词外是否有首字母大写的单词来粗略近似专有名词
    PROPER_NOUN_PATTERN = re.compile(r'\b[A-Z][a-z]+\b')

    def extract_claims(self, text: str, section_name: str) -> List[Claim]:
        sentences = self._split_sentences(text)
        claims = []

        for sent in sentences:
            if self._is_filler(sent) or self._is_subjective(sent) or self._is_conditional(sent) or self._is_insufficient(sent):
                continue

            cited_sources = self.CITATION_PATTERN.findall(sent)
            if not cited_sources:
                cited_sources = ["no_citation"]
            else:
                cited_sources = [s.strip() for s in cited_sources]

            numbers = self.NUMBER_PATTERN.findall(sent)
            dates = self.DATE_PATTERN.findall(sent)

            # 清理句子中的引用标记（留纯文本）
            clean_text = self.CITATION_PATTERN.sub('', sent).strip()

            if len(clean_text) < 15:
                continue

            # 计算 specificity_score
            score = 0.0
            if numbers:
                score += 0.3
            if dates:
                score += 0.2
            if "no_citation" not in cited_sources:
                score += 0.3
                
            # 简易专有名词检查 (去掉首词的干扰)
            words = clean_text.split()
            if len(words) > 1:
                tail = " ".join(words[1:])
                if self.PROPER_NOUN_PATTERN.search(tail):
                    score += 0.2

            if score <= 0.3:
                continue  # 抛弃不够具体的句子

            claim_id = hashlib.md5(clean_text.encode()).hexdigest()[:10]

            claims.append(Claim(
                claim_id=claim_id,
                text=clean_text,
                section=section_name,
                cited_sources=cited_sources,
                contains_numbers=len(numbers) > 0,
                extracted_numbers=numbers,
                specificity_score=round(score, 2)
            ))

        return claims

    def _split_sentences(self, text: str) -> List[str]:
        # 按句号、问号、感叹号分割，但不在缩写处分割，支持换行符分割
        text = text.replace('\n', ' ')
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_filler(self, sentence: str) -> bool:
        """过滤过渡句和连接句"""
        fillers = [
            "in summary", "overall", "to summarize", "in conclusion",
            "the following section", "this report", "firstly", "secondly", 
            "it is worth noting", "as mentioned", "as seen"
        ]
        lower = sentence.lower().strip()
        return any(lower.startswith(f) or lower == f for f in fillers)

    def _is_subjective(self, sentence: str) -> bool:
        """过滤主观定性评价"""
        subjective_keywords = [
            "is a prominent", "is an excellent", "is widely considered",
            "is a great", "is amazing", "we believe", "it seems",
            "is highly regarded"
        ]
        lower = sentence.lower()
        return any(k in lower for k in subjective_keywords)

    def _is_conditional(self, sentence: str) -> bool:
        """过滤条件句/假设句"""
        lower = sentence.lower()
        if lower.startswith("if ") or "would be" in lower or "could potentially" in lower:
            return True
        return False

    def _is_insufficient(self, sentence: str) -> bool:
        """过滤声明证据不足的套话"""
        lower = sentence.lower()
        insufficient_phrases = [
            "evidence is insufficient", "not provided in the", "no distinct information",
            "does not mention", "cannot be determined", "unable to verify"
        ]
        return any(p in lower for p in insufficient_phrases)
