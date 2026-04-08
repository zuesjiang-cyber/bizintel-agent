"""
报告生成器

职责：
- 将各步骤分析结果组装成完整 memo
- 生成 executive summary
- 格式化来源列表
- 集成验证结果
"""

import re
from typing import Any, Dict, List

from agent.config import settings
from agent.llm_utils import (
    build_openai_client,
    build_stub_executive_summary,
    generate_text_response,
    should_use_stub_llm,
)
from agent.prompts.research import (
    EVIDENCE_WRITING_SYSTEM_PROMPT,
    STRICT_VERIFICATION_SYSTEM_PROMPT,
    build_evidence_writing_prompt,
    build_strict_verification_prompt,
)
from agent.schemas import (
    AnalysisMode, AnalysisPlan, GeneratedMemo, MemoSection,
    ConfidenceLevel,
    DocumentMeta,
)
from verification.claim_extractor import ClaimExtractor
from verification.evidence_verifier import EvidenceVerifier


class ReportWriter:
    def __init__(self, demo_mode: bool = False, enable_report_verification: bool = True):
        self.client = None
        self.demo_mode = demo_mode
        self.enable_report_verification = enable_report_verification
        self.claim_extractor = ClaimExtractor() if enable_report_verification else None
        self.use_stub_llm = demo_mode or should_use_stub_llm(settings.llm_mode, settings.openai_api_key)
        self.verifier = EvidenceVerifier(use_dummy_model=self.use_stub_llm) if enable_report_verification else None
        
    def _get_client(self):
        if self.client is None:
            self.client = build_openai_client(settings.openai_api_key, settings.openai_api_base)
        return self.client

    def generate_memo(
        self,
        plan: AnalysisPlan,
        step_outputs: Dict[str, Any],
        sources: List[DocumentMeta],
    ) -> GeneratedMemo:
        # 1. 收集所有分析内容
        sections = []
        all_evidence = {}  # source_id -> [evidence chunks]
        source_meta_by_id = {source.source_id: source for source in sources}

        for step_name, output in step_outputs.items():
            if isinstance(output, dict) and "content" in output:
                section_contract = plan.contract.get("sections", {}).get(step_name, {})
                content = self._compose_section_content(step_name, output, section_contract)
                section = MemoSection(
                    title=step_name,
                    content=content,
                    evidence_notes=output.get("evidence_notes", []),
                    writing_trace=output.get("writing_trace", {}),
                )
                sections.append(section)

                # 收集 evidence
                for ev in output.get("raw_evidence", []):
                    sid = ev["source_id"]
                    if sid not in all_evidence:
                        all_evidence[sid] = []
                    meta = source_meta_by_id.get(sid)
                    enriched = dict(ev)
                    if meta is not None:
                        enriched["source_type"] = meta.source_type
                        enriched["is_primary"] = meta.is_primary
                    all_evidence[sid].append(enriched)

        # 2. 提取 claims 并验证
        if self.enable_report_verification:
            for section in sections:
                section.content, section.claims, section.verification_results = self._verify_and_gate_section(
                    section=section,
                    evidence_store=all_evidence,
                )

        # 3. 生成 executive summary
        all_content = "\n\n".join(
            f"### {s.title}\n{s.content}" for s in sections
        )
        executive_summary = self._generate_executive_summary(all_content) if all_content else "No analysis content available."

        # 4. 组装 memo
        memo = GeneratedMemo(
            title=self._generate_title(plan),
            mode=plan.mode,
            query=plan.user_query,
            executive_summary=executive_summary,
            contract=plan.contract,
            sections=sections,
            sources=sources,
        )

        # 5. 计算总体置信度
        all_verifications = []
        for s in sections:
            all_verifications.extend(s.verification_results)

        if self.enable_report_verification and all_verifications:
            stats = self.verifier.summary_stats(all_verifications)
            memo.overall_confidence = stats["verified_claim_coverage"]

        return memo

    def _compose_section_content(self, step_name: str, output: Dict[str, Any], section_contract: dict) -> str:
        evidence_notes = output.get("evidence_notes") or []
        existing_content = output.get("content", "")
        fallback_content = self._build_stub_section_from_notes(
            step_name,
            evidence_notes,
            section_contract,
            output.get("missing_facts", []),
        )
        if not evidence_notes and existing_content:
            output.setdefault("writing_trace", {"draft": existing_content, "critique": [], "final_answer": existing_content})
            return existing_content
        if self.use_stub_llm:
            output["writing_trace"] = {"draft": fallback_content, "critique": [], "final_answer": fallback_content}
            return fallback_content

        prompt = build_evidence_writing_prompt(
            mode="memo_section",
            step_name=step_name,
            section_contract=section_contract,
            draft_notes=existing_content[:500],
            evidence_notes=evidence_notes[:2],
            previous_findings="",
        )
        client = self._get_client()
        try:
            response = generate_text_response(
                client,
                model=settings.openai_model,
                system_prompt=EVIDENCE_WRITING_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=350,
                temperature=0.1,
                max_retries=1,
            )
        except Exception:
            output["writing_trace"] = {"draft": existing_content, "critique": ["llm_write_failed"], "final_answer": fallback_content}
            return fallback_content
        payload = self._parse_json_payload_or_none(response)
        normalized_content = self._normalize_llm_section_response(payload, response)
        if normalized_content and self._is_valid_section_content(normalized_content, evidence_notes):
            output["writing_trace"] = {
                "draft": payload.get("draft", "") if isinstance(payload, dict) else response,
                "critique": payload.get("critique", []) if isinstance(payload, dict) else [],
                "final_answer": normalized_content,
            }
            return normalized_content
        critique = payload.get("critique", []) if isinstance(payload, dict) else []
        output["writing_trace"] = {
            "draft": payload.get("draft", response) if isinstance(payload, dict) else response,
            "critique": [*critique, "invalid_or_uncited_final_answer"],
            "final_answer": fallback_content,
        }
        return fallback_content

    def _build_stub_section_from_notes(
        self,
        step_name: str,
        evidence_notes: List[dict],
        section_contract: dict,
        missing_facts: List[str],
    ) -> str:
        if not evidence_notes:
            return "Insufficient evidence in the provided source pack to support this section."

        fact_lines = []
        commentary_lines = []
        inference_lines = []
        for note in evidence_notes[:8]:
            sentence = self._best_supported_sentence(note)
            if not sentence:
                continue
            line = f"- {sentence} [Chunk: {note['chunk_id']}] [Source: {note['source_id']}]"
            if note["evidence_type"] == "management_commentary":
                commentary_lines.append(line)
            elif note["evidence_type"] == "inference":
                inference_lines.append(line)
            else:
                fact_lines.append(line)

        lines: List[str] = []
        if fact_lines:
            lines.extend(fact_lines[:3])
        if commentary_lines:
            lines.append("")
            lines.append("Management commentary:")
            lines.extend(commentary_lines[:2])
        if inference_lines and not section_contract.get("fact_first", False):
            lines.append("")
            lines.append("Inference:")
            lines.extend(inference_lines[:1])
        if missing_facts:
            lines.append("")
            lines.append(f"- Insufficient evidence to directly verify: {', '.join(missing_facts[:2])}.")
        if not lines:
            return "Insufficient evidence in the provided source pack to support this section."
        return "\n".join(lines).strip()

    def _best_supported_sentence(self, note: dict) -> str:
        complete_evidence = self._first_complete_evidence_sentence(note.get("evidence_text", ""))
        if complete_evidence:
            return complete_evidence
        candidates = [
            self._clean_sentence_fragment(note.get("claim", "")),
            self._clean_sentence_fragment(note.get("evidence_text", "")),
        ]
        for candidate in candidates:
            if self._looks_like_complete_sentence(candidate) or self._looks_like_stub_evidence_sentence(candidate):
                return candidate
        return ""

    def _first_complete_evidence_sentence(self, text: str) -> str:
        normalized = re.sub(r"\s+", " ", (text or "")).strip()
        if not normalized:
            return ""
        sentence_candidates = re.split(r"(?<=[.!?])\s+", normalized)
        for candidate in sentence_candidates:
            if "..." in candidate:
                continue
            cleaned = self._clean_sentence_fragment(candidate)
            if self._looks_like_stub_evidence_sentence(cleaned):
                return cleaned
        return ""

    def _looks_like_stub_evidence_sentence(self, text: str) -> bool:
        words = re.findall(r"\b\w+\b", text)
        if len(words) < 5:
            return False
        if text[:1].islower():
            return False
        if "..." in text:
            return False
        if text.endswith(":"):
            return False
        return True

    def _clean_sentence_fragment(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "")).strip(" -\"'")
        if not cleaned:
            return ""
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" -\"'")
        if cleaned.startswith("•"):
            cleaned = cleaned.lstrip("•").strip()
        if not cleaned.endswith((".", "!", "?")):
            cleaned += "."
        return cleaned

    def _looks_like_complete_sentence(self, text: str) -> bool:
        words = re.findall(r"\b\w+\b", text)
        if len(words) < 6:
            return False
        if text[:1].islower():
            return False
        lowered = text.lower()
        verb_markers = {
            "is", "are", "was", "were", "has", "have", "grew", "grow", "drives", "serve",
            "serves", "use", "uses", "includes", "include", "provides", "providing",
            "reported", "reports", "generated", "increased", "decreased", "expects",
        }
        return any(marker in lowered.split() for marker in verb_markers) or any(char.isdigit() for char in text)

    def _normalize_llm_section_response(self, payload: dict | None, raw_response: str) -> str:
        if isinstance(payload, dict):
            normalized = self._normalize_generated_answer(payload.get("final_answer"))
            if normalized:
                return normalized
        return self._normalize_generated_answer(raw_response)

    def _normalize_generated_answer(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            lines = []
            for item in value:
                normalized = self._normalize_generated_answer(item)
                if not normalized:
                    continue
                if normalized.startswith("- ") or normalized.startswith("##"):
                    lines.append(normalized)
                else:
                    lines.append(f"- {normalized}")
            return "\n".join(lines).strip()
        if isinstance(value, dict):
            label_map = {
                "facts": "Facts",
                "reported_facts": "Facts",
                "management_commentary": "Management Commentary",
                "commentary": "Management Commentary",
                "inference": "Inference",
                "uncertainty": "Uncertainty",
            }
            blocks = []
            seen_labels = set()
            for key in ("facts", "reported_facts", "management_commentary", "commentary", "inference", "uncertainty"):
                if key not in value:
                    continue
                label = label_map[key]
                if label in seen_labels:
                    continue
                rendered = self._render_generated_block(label, value[key])
                if rendered:
                    blocks.append(rendered)
                    seen_labels.add(label)
            for key, block_value in value.items():
                if key in {"facts", "reported_facts", "management_commentary", "commentary", "inference", "uncertainty"}:
                    continue
                rendered = self._render_generated_block(key.replace("_", " ").title(), block_value)
                if rendered:
                    blocks.append(rendered)
            return "\n\n".join(blocks).strip()
        return ""

    def _render_generated_block(self, label: str, value: Any) -> str:
        normalized = self._normalize_generated_answer(value)
        if not normalized:
            return ""
        if normalized.startswith("- ") and "\n" not in normalized:
            return f"**{label}**\n\n{normalized}"
        if normalized.startswith("- ") or normalized.startswith("##") or "\n- " in normalized:
            return f"**{label}**\n\n{normalized}"
        return f"**{label}**\n\n- {normalized}"

    def _is_valid_section_content(self, content: str, evidence_notes: List[dict]) -> bool:
        stripped = (content or "").strip()
        if not stripped:
            return False
        if stripped.startswith("{") or stripped.startswith("["):
            return False
        if evidence_notes and "[Source:" not in stripped:
            return "insufficient evidence" in stripped.lower()
        return True

    def _verify_and_gate_section(self, section: MemoSection, evidence_store: Dict[str, List[dict]]) -> tuple[str, List, List]:
        claims = self.claim_extractor.extract_claims(section.content, section.title)
        if not claims:
            return section.content, [], []

        verification_results = self.verifier.verify_memo(claims, evidence_store)
        gated_content = self._apply_verification_gating(section.content, verification_results)
        if gated_content == section.content:
            return section.content, claims, verification_results

        gated_claims = self.claim_extractor.extract_claims(gated_content, section.title)
        if not gated_claims and all(
            self._verification_action(result) == "downgrade"
            for result in verification_results
        ) and any(result.claim.contains_numbers for result in verification_results):
            return section.content, claims, verification_results
        gated_results = self.verifier.verify_memo(gated_claims, evidence_store) if gated_claims else []
        return gated_content, gated_claims, gated_results

    def _rewrite_after_verification(self, section_title: str, content: str, verification_results: List) -> str:
        flagged = [
            {
                "claim_text": result.claim.text,
                "confidence": result.confidence.value,
                "failure_reason": result.failure_reason,
                "failure_stage": result.failure_stage,
                "contains_numbers": result.claim.contains_numbers,
                "risk_level": result.claim.risk_level,
                "supporting_source_ids": result.supporting_source_ids,
            }
            for result in verification_results
            if self._verification_action(result) != "keep"
        ]
        if not flagged:
            return content
        try:
            prompt = build_strict_verification_prompt(
                section_title=section_title,
                content=content,
                verification_findings=flagged,
            )
            rewritten = generate_text_response(
                self._get_client(),
                model=settings.openai_model,
                system_prompt=STRICT_VERIFICATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                max_tokens=900,
                temperature=0.1,
            ).strip()
            payload = self._parse_json_payload_or_none(rewritten)
            if isinstance(payload, dict) and isinstance(payload.get("rewritten_section"), str):
                return payload["rewritten_section"].strip() or content
            return rewritten or content
        except Exception:
            return content

    def render_memo(self, memo: GeneratedMemo) -> str:
        """将 GeneratedMemo 渲染为 Markdown 文本"""
        # 生成 sources 列表
        sources_list = self._format_sources(memo.sources)

        # 生成 verification summary
        verification_summary = self._format_verification_summary(memo)

        # 生成 executive summary
        all_content = "\n".join(s.content for s in memo.sections)
        exec_summary = memo.executive_summary or self._generate_executive_summary(all_content)

        # 填充模板
        # 根据模式用不同的模板字段
        filled = f"# {memo.title}\n\n"
        filled += f"Generated: {memo.generated_at[:10]} | "
        filled += f"Mode: {memo.mode.value} | "
        if self.enable_report_verification:
            confidence_label = "Demo Heuristic Support" if self.demo_mode else "Confidence"
            filled += f"{confidence_label}: {memo.overall_confidence:.0%}"
        filled += "\n\n"
        if self.demo_mode and self.enable_report_verification:
            filled += (
                "> Offline demo mode uses deterministic stub summarization and a lightweight heuristic verifier. "
                "Treat these support metrics as walkthrough aids, not analyst-grade confidence scores.\n\n"
            )
        filled += f"## Executive Summary\n{exec_summary}\n\n"
        if memo.contract.get("required_slots"):
            filled += "## Coverage Contract\n"
            for slot in memo.contract["required_slots"]:
                filled += f"- {slot}\n"
            filled += "\n"

        for section in memo.sections:
            filled += f"## {self._format_section_title(section.title)}\n"
            filled += f"{section.content}\n\n"

            # 加置信度标注
            if self.enable_report_verification and section.verification_results:
                stats = self.verifier.summary_stats(section.verification_results)
                section_label = "Offline demo heuristic support" if self.demo_mode else "Verified section support"
                filled += f"*{section_label}: {stats['verified_claim_coverage']:.0%} "
                filled += "claims supported*\n\n"

        filled += f"---\n## Sources\n{sources_list}\n"
        if self.enable_report_verification:
            filled += f"\n## Verification Summary\n{verification_summary}\n"

        return filled

    def _generate_executive_summary(self, all_content: str) -> str:
        return build_stub_executive_summary([all_content])

    def _apply_verification_gating(self, content: str, verification_results: List) -> str:
        gated_results = [result for result in verification_results if self._verification_action(result) != "keep"]
        if not gated_results:
            return content

        uses_line_layout = "\n" in content
        segments = content.splitlines() if uses_line_layout else re.split(r'(?<=[.!?])\s+', content)
        kept_segments = []
        for segment in segments:
            stripped = segment.strip()
            if not stripped:
                if uses_line_layout:
                    kept_segments.append(segment)
                continue

            normalized = re.sub(r'\[Source:\s*[^\]]+\]', '', stripped)
            normalized = re.sub(r'\[Chunk:\s*[^\]]+\]', '', normalized)
            normalized = normalized.lstrip("-• ").strip()
            matched_result = next(
                (
                    result for result in gated_results
                    if result.claim.text in normalized or normalized in result.claim.text
                ),
                None,
            )
            if matched_result is None:
                kept_segments.append(segment)
                continue

            action = self._verification_action(matched_result)
            if action == "delete":
                continue

            replacement = "- Insufficient evidence in the source pack to verify this point directly."
            if replacement not in kept_segments:
                kept_segments.append(replacement)

        if not kept_segments:
            return "Insufficient evidence in the provided source pack to support this section."

        return "\n".join(kept_segments).strip() if uses_line_layout else " ".join(kept_segments).strip()

    def _generate_title(self, plan: AnalysisPlan) -> str:
        mode_labels = {
            AnalysisMode.COMPANY: "Company Research Memo",
            AnalysisMode.INDUSTRY: "Industry Landscape Report",
            AnalysisMode.COMPETITIVE: "Competitive Analysis",
        }
        label = mode_labels.get(plan.mode, "Research Memo")
        return f"{label}: {plan.user_query}"

    def _format_sources(self, sources: List[DocumentMeta]) -> str:
        lines = []
        for i, source in enumerate(sources, 1):
            title = source.title
            # Completely strip any \n literals or actual newlines that might be in the title
            title = title.replace("\\n", " ").replace("\n", " ")
            title = " ".join(title.split()) # compress multiple spaces
            
            line = f"- **[{source.source_id}]** {title}"
            if source.url:
                clean_url = source.url.split("\\n", 1)[0].split("\n", 1)[0].strip()
                line += f" — [Link]({clean_url})"
            lines.append(line)
        return "\n".join(lines)

    def _format_verification_summary(self, memo: GeneratedMemo) -> str:
        all_results = []
        for section in memo.sections:
            all_results.extend(section.verification_results)

        if not all_results:
            return "No claims verified."

        stats = self.verifier.summary_stats(all_results)
        
        coverage_label = "Heuristic support rate" if self.demo_mode else "Verified claim coverage"
        score_label = "Average heuristic score" if self.demo_mode else "Average NLI score"
        summary_text = (
            f"- Total claims analyzed: {stats['total_claims']}\n"
            f"- Strong support: {stats['strong']}\n"
            f"- Moderate support: {stats['moderate']}\n"
            f"- Weak support: {stats['weak']}\n"
            f"- Unsupported: {stats['unsupported']}\n"
            f"- Contradicted: {stats.get('contradicted', 0)}\n"
            f"- {coverage_label}: {stats['verified_claim_coverage']:.0%}\n"
            f"- {score_label}: {stats['avg_nli_score']:.2f}\n"
        )
        
        # specific weak/unsupported claims listing
        flagged_claims = [r for r in all_results if r.confidence in [ConfidenceLevel.WEAK, ConfidenceLevel.UNSUPPORTED]]
        
        if flagged_claims:
            summary_text += "\n### ⚠️ Flagged Claims (Weak / Unsupported)\n"
            for r in flagged_claims:
                reason = "No matching evidence" if r.confidence == ConfidenceLevel.UNSUPPORTED else "Evidence contradicts or weakly supports"
                summary_text += f"- **Claim:** \"{r.claim.text}\"\n"
                summary_text += f"  - *Score:* {r.nli_score:.2f} ({r.confidence.value}) — {reason}\n"

        return summary_text

    def _format_section_title(self, name: str) -> str:
        return name.replace("_", " ").title()

    def _parse_json_payload_or_none(self, content: str):
        stripped = (content or "").strip()
        if not stripped:
            return None
        match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
        if match:
            stripped = match.group(1)
        try:
            import json

            return json.loads(stripped)
        except Exception:
            return None

    def _verification_action(self, result: Any) -> str:
        hard_delete_reasons = {
            "missing_citation",
            "bad_source_id",
            "numeric_mismatch",
            "period_mismatch",
            "currency_mismatch",
            "unit_mismatch",
            "direction_mismatch",
            "primary_source_missing",
            "contradiction",
        }
        if result.confidence == ConfidenceLevel.CONTRADICTED:
            return "delete"
        if result.failure_reason in hard_delete_reasons:
            return "delete"
        if result.claim.contains_numbers and result.confidence == ConfidenceLevel.UNSUPPORTED:
            return "delete"
        if result.failure_reason == "low_entailment" or result.confidence in {ConfidenceLevel.WEAK, ConfidenceLevel.UNSUPPORTED}:
            return "downgrade"
        return "keep"
