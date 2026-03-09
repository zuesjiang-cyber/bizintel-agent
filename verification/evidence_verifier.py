"""
验证从 memo 提取出的 claim 是否有原文支撑。
"""

import logging
from typing import Dict, List, Optional
import numpy as np

from sentence_transformers import CrossEncoder

from agent.config import settings
from agent.schemas import Claim, ConfidenceLevel, VerificationResult

logger = logging.getLogger(__name__)


class EvidenceVerifier:
    def __init__(self):
        self._nli_model: Optional[CrossEncoder] = None
        self.strong_threshold = settings.entailment_strong_threshold
        self.moderate_threshold = settings.entailment_moderate_threshold

    def _get_model(self) -> CrossEncoder:
        if self._nli_model is None:
            logger.info("Loading NLI model for verification...")
            self._nli_model = CrossEncoder(settings.nli_model)
        return self._nli_model

    def verify_memo(self, claims: List[Claim], evidence_store: Dict[str, List[str]]) -> List[VerificationResult]:
        """
        验证一组 claim。
        evidence_store: source_id -> [text_chunks]
        """
        if not claims:
            return []

        results = []
        pairs_to_score = []
        claim_evidence_map = []  # 记录每个 pair 对应的 (claim_idx, evidence_text)

        # 1. 准备待打分的 (evidence, claim) 必须对
        # NLI 模型的输入顺序通常是 (Premise, Hypothesis) = (Evidence, Claim)
        for i, claim in enumerate(claims):
            has_valid_source = False
            for source_id in claim.cited_sources:
                if source_id in evidence_store:
                    has_valid_source = True
                    for chunk in evidence_store[source_id]:
                        pairs_to_score.append((chunk, claim.text))
                        claim_evidence_map.append((i, chunk))

            if not has_valid_source:
                # 没有任何有效来源，直接判无支撑
                results.append(VerificationResult(
                    claim=claim,
                    confidence=ConfidenceLevel.UNSUPPORTED,
                    nli_score=0.0,
                    numeric_verified=False if claim.contains_numbers else None,
                    supporting_evidence=[],
                    explanation="No valid sources cited or source text not found."
                ))

        if not pairs_to_score:
            return results

        # 2. 批量跑 NLI 模型
        # NLI 输出通常是 3 分类：[Contradiction, Entailment, Neutral]
        # 这里用的是 cross-encoder/nli-deberta-v3-base，输出为 [Contradiction, Entailment, Neutral]
        logger.info(f"Running NLI verification for {len(pairs_to_score)} pairs...")
        model = self._get_model()
        scores = model.predict(pairs_to_score, show_progress_bar=False)

        # 提取 Entailment (蕴含) 的分数。对于 deberta-v3-base，Entailment 在 index 1
        # 但我们为了稳妥，先转成 prob，取 index 1
        import torch
        probs = torch.nn.functional.softmax(torch.tensor(scores), dim=1).numpy()
        entailment_scores = probs[:, 1]

        # 3. 聚合结果
        # 一个 claim 可能对应多个 evidence chunk，取最高分
        best_scores = {i: 0.0 for i in range(len(claims))}
        best_evidence = {i: "" for i in range(len(claims))}

        for pair_idx, (claim_idx, chunk) in enumerate(claim_evidence_map):
            score = float(entailment_scores[pair_idx])
            if score > best_scores[claim_idx]:
                best_scores[claim_idx] = score
                best_evidence[claim_idx] = chunk

        # 4. 生成最终判定
        # 需要注意的是，我们之前可能已经提前把没有 source 的 claim 放进了 results。
        # 为了保持顺序，我们用一个临时字典
        temp_results = {}
        for r in results:
            temp_results[r.claim.claim_id] = r

        for i, claim in enumerate(claims):
            if claim.claim_id in temp_results:
                continue

            score = best_scores[i]
            evidence = best_evidence[i]

            # 规则 1：检查数字是否匹配
            numeric_verified = None
            if claim.contains_numbers:
                numeric_verified = self._verify_numbers(claim.extracted_numbers, evidence)

            # 规则 2：综合 NLI 和规则得出最终置信度
            confidence = ConfidenceLevel.WEAK
            if score >= self.strong_threshold:
                confidence = ConfidenceLevel.STRONG
            elif score >= self.moderate_threshold:
                confidence = ConfidenceLevel.MODERATE

            # 如果数字校验失败，降级
            if numeric_verified is False:
                if confidence == ConfidenceLevel.STRONG:
                    confidence = ConfidenceLevel.MODERATE
                elif confidence == ConfidenceLevel.MODERATE:
                    confidence = ConfidenceLevel.WEAK

            explanation = (
                f"NLI Entailment Score: {score:.2f}. "
                f"Numeric match: {'Yes' if numeric_verified else ('No' if numeric_verified is False else 'N/A')}."
            )

            res = VerificationResult(
                claim=claim,
                confidence=confidence,
                nli_score=score,
                numeric_verified=numeric_verified,
                supporting_evidence=[evidence] if evidence else [],
                explanation=explanation
            )
            temp_results[claim.claim_id] = res

        # 按原顺序返回
        final_results = []
        for claim in claims:
            final_results.append(temp_results[claim.claim_id])

        return final_results

    def _verify_numbers(self, extracted_numbers: List[str], evidence_text: str) -> bool:
        """简单的数字匹配：提取的数字必须（部分）出现在 evidence 中"""
        if not evidence_text:
            return False

        # 简化版：这里只要求至少有一个关键数字在原文中能找到。
        # 实际工业界会用专门的 NER 或 LLM 做更细致的比对。
        for num in extracted_numbers:
            # 简单去除 $ 和逗号进行检查
            clean_num = num.replace("$", "").replace(",", "").strip()
            # 常见单位如 B, M 需要原样保留去匹配，或者提取数字本体校验
            # 这里做最简单的暴力匹配
            if clean_num in evidence_text or num in evidence_text:
                return True
        return False

    def summary_stats(self, results: List[VerificationResult]) -> dict:
        """生成验证统计信息"""
        total = len(results)
        if total == 0:
            return {
                "total_claims": 0,
                "strong": 0, "moderate": 0, "weak": 0, "unsupported": 0,
                "citation_coverage": 0.0, "avg_nli_score": 0.0
            }

        counts = {c: 0 for c in ConfidenceLevel}
        for r in results:
            counts[r.confidence] += 1

        supported = total - counts[ConfidenceLevel.UNSUPPORTED]

        return {
            "total_claims": total,
            "strong": counts[ConfidenceLevel.STRONG],
            "moderate": counts[ConfidenceLevel.MODERATE],
            "weak": counts[ConfidenceLevel.WEAK],
            "unsupported": counts[ConfidenceLevel.UNSUPPORTED],
            "citation_coverage": supported / total if total > 0 else 0.0,
            "avg_nli_score": sum(r.nli_score for r in results) / total
        }
