"""
报告生成器

职责：
- 将各步骤分析结果组装成完整 memo
- 生成 executive summary
- 格式化来源列表
- 集成验证结果
"""

from datetime import datetime
from typing import Any, Dict, List

from openai import OpenAI

from agent.config import settings
from agent.schemas import (
    AnalysisMode, AnalysisPlan, GeneratedMemo, MemoSection,
    DocumentMeta, VerificationResult,
)
from agent.prompts.synthesis import (
    MEMO_SYSTEM_PROMPT, COMPANY_MEMO_TEMPLATE, INDUSTRY_MEMO_TEMPLATE,
    COMPETITIVE_MEMO_TEMPLATE, EXECUTIVE_SUMMARY_PROMPT,
)
from verification.claim_extractor import ClaimExtractor
from verification.evidence_verifier import EvidenceVerifier


class ReportWriter:
    def __init__(self):
        self.client = None
        self.claim_extractor = ClaimExtractor()
        self.verifier = EvidenceVerifier()
        
    def _get_client(self):
        if self.client is None:
            self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_api_base)
        return self.client

    def generate_memo(
        self,
        plan: AnalysisPlan,
        step_outputs: Dict[str, Any],
        sources: List[DocumentMeta],
    ) -> GeneratedMemo:
        # 1. 收集所有分析内容
        sections = []
        all_evidence = {}  # source_id -> [text_chunks]

        for step_name, output in step_outputs.items():
            if isinstance(output, dict) and "content" in output:
                content = output["content"]
                section = MemoSection(title=step_name, content=content)
                sections.append(section)

                # 收集 evidence
                for ev in output.get("raw_evidence", []):
                    sid = ev["source_id"]
                    if sid not in all_evidence:
                        all_evidence[sid] = []
                    all_evidence[sid].append(ev["text"])

        # 2. 提取 claims 并验证
        for section in sections:
            claims = self.claim_extractor.extract_claims(
                section.content, section.title
            )
            section.claims = claims

            if claims:
                verification_results = self.verifier.verify_memo(claims, all_evidence)
                section.verification_results = verification_results

        # 3. 生成 executive summary
        all_content = "\n\n".join(
            f"### {s.title}\n{s.content}" for s in sections
        )
        # 4. 组装 memo
        memo = GeneratedMemo(
            title=self._generate_title(plan),
            mode=plan.mode,
            query=plan.user_query,
            sections=sections,
            sources=sources,
        )

        # 5. 计算总体置信度
        all_verifications = []
        for s in sections:
            all_verifications.extend(s.verification_results)

        if all_verifications:
            stats = self.verifier.summary_stats(all_verifications)
            memo.overall_confidence = stats["citation_coverage"]

        return memo

    def render_memo(self, memo: GeneratedMemo) -> str:
        """将 GeneratedMemo 渲染为 Markdown 文本"""
        # 选择模板
        template_map = {
            AnalysisMode.COMPANY: COMPANY_MEMO_TEMPLATE,
            AnalysisMode.INDUSTRY: INDUSTRY_MEMO_TEMPLATE,
            AnalysisMode.COMPETITIVE: COMPETITIVE_MEMO_TEMPLATE,
        }
        template = template_map.get(memo.mode, COMPANY_MEMO_TEMPLATE)

        # 构建各 section 内容
        section_contents = {}
        for section in memo.sections:
            section_contents[section.title] = section.content

        # 生成 sources 列表
        sources_list = self._format_sources(memo.sources)

        # 生成 verification summary
        verification_summary = self._format_verification_summary(memo)

        # 生成 executive summary
        all_content = "\n".join(s.content for s in memo.sections)
        exec_summary = self._generate_executive_summary(all_content)

        # 填充模板
        # 根据模式用不同的模板字段
        filled = f"# {memo.title}\n\n"
        filled += f"Generated: {memo.generated_at[:10]} | "
        filled += f"Mode: {memo.mode.value} | "
        filled += f"Confidence: {memo.overall_confidence:.0%}\n\n"
        filled += f"## Executive Summary\n{exec_summary}\n\n"

        for section in memo.sections:
            filled += f"## {self._format_section_title(section.title)}\n"
            filled += f"{section.content}\n\n"

            # 加置信度标注
            if section.verification_results:
                stats = self.verifier.summary_stats(section.verification_results)
                filled += f"*Section confidence: {stats['citation_coverage']:.0%} "
                filled += f"claims supported*\n\n"

        filled += f"---\n## Sources\n{sources_list}\n\n"
        filled += f"## Verification Summary\n{verification_summary}\n"

        return filled

    def _generate_executive_summary(self, all_content: str) -> str:
        enhanced_prompt = (
            f"Based on the following analysis sections, generate a comprehensive Executive Summary.\n\n"
            f"Constraints & Requirements:\n"
            f"- Write 3-5 distinct paragraphs (approx 200-400 words total).\n"
            f"- Cover these core aspects: Key Findings, Financial/Funding Highlights, Competitive Positioning, and Future Outlook/Risks.\n"
            f"- Output MUST be written as a cohesive narrative summary, not just bullet points.\n\n"
            f"Evidence text to summarize:\n{all_content[:4000]}"
        )

        client = self._get_client()
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": MEMO_SYSTEM_PROMPT},
                {"role": "user", "content": enhanced_prompt},
            ],
            max_tokens=800,
            temperature=0.2,
        )
        summary = response.choices[0].message.content.strip()
        # Fallback if the model stops generating mid-sentence
        if not summary.endswith((".", "!", "?")):
            summary += "..."
        return summary

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
                line += f" — [Link]({source.url})"
            lines.append(line)
        return "\n".join(lines)

    def _format_verification_summary(self, memo: GeneratedMemo) -> str:
        from agent.schemas import ConfidenceLevel
        
        all_results = []
        for section in memo.sections:
            all_results.extend(section.verification_results)

        if not all_results:
            return "No claims verified."

        stats = self.verifier.summary_stats(all_results)
        
        summary_text = (
            f"- Total claims analyzed: {stats['total_claims']}\n"
            f"- Strong support: {stats['strong']}\n"
            f"- Moderate support: {stats['moderate']}\n"
            f"- Weak support: {stats['weak']}\n"
            f"- Unsupported: {stats['unsupported']}\n"
            f"- Citation coverage: {stats['citation_coverage']:.0%}\n"
            f"- Average NLI score: {stats['avg_nli_score']:.2f}\n"
        )
        
        # specific weak/unsupported claims listing
        flagged_claims = [r for r in all_results if r.confidence in [ConfidenceLevel.WEAK, ConfidenceLevel.UNSUPPORTED]]
        
        if flagged_claims:
            summary_text += "\n### ⚠️ Flagged Claims (Weak / Unsupported)\n"
            for r in flagged_claims:
                reason = "No matching evidence" if r.confidence == ConfidenceLevel.UNSUPPORTED else "Evidence contradicts or weakly supports"
                summary_text += f"- **Claim:** \"{r.claim.text}\"\n"
                summary_text += f"  - *Score:* {r.score:.2f} ({r.confidence.value}) — {reason}\n"

        return summary_text

    def _format_section_title(self, name: str) -> str:
        return name.replace("_", " ").title()
