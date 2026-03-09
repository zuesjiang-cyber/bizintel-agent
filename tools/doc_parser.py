"""
文档解析与统一格式化

职责：
- 读取 company_pack 目录下的各种格式文件
- 统一输出 DocumentMeta + 原始文本
- 不做分 chunk，分 chunk 是 retrieval/chunking.py 的事
"""

import json
from pathlib import Path
from typing import List, Tuple

from agent.schemas import DocumentMeta


def load_company_pack(company_dir: Path) -> List[Tuple[DocumentMeta, str]]:
    """
    读取一个公司的数据包，返回 [(meta, raw_text), ...]
    """
    documents = []

    # 加载 profile.json
    profile_path = company_dir / "profile.json"
    if profile_path.exists():
        with open(profile_path) as f:
            profile = json.load(f)
        meta = DocumentMeta(
            source_id=f"{company_dir.name}_profile",
            source_type="profile",
            title=f"{profile.get('company_name', company_dir.name)} - Company Profile",
            company=profile.get("company_name", company_dir.name),
        )
        # 将 JSON 转为可读文本
        text = _profile_to_text(profile)
        documents.append((meta, text))

    # 加载 .txt 文件
    for txt_file in sorted(company_dir.glob("*.txt")):
        raw = txt_file.read_text(encoding="utf-8")
        source_url, date, title, body = _parse_txt_header(raw)

        source_type = _classify_source_type(txt_file.name)
        meta = DocumentMeta(
            source_id=f"{company_dir.name}_{txt_file.stem}",
            source_type=source_type,
            title=title or txt_file.stem,
            url=source_url,
            date=date,
            company=company_dir.name,
        )
        documents.append((meta, body))

    # 加载 PDF（如果有）
    for pdf_file in company_dir.glob("*.pdf"):
        meta, text = _parse_pdf(pdf_file, company_dir.name)
        documents.append((meta, text))

    return documents


def _profile_to_text(profile: dict) -> str:
    """将 profile JSON 转为可读文本段落"""
    lines = []
    lines.append(f"Company: {profile.get('company_name', 'Unknown')}")
    lines.append(f"Founded: {profile.get('founded', 'Unknown')}")
    lines.append(f"Headquarters: {profile.get('headquarters', 'Unknown')}")
    lines.append(f"Founders: {', '.join(profile.get('founders', []))}")
    lines.append(f"Employees: {profile.get('employee_count', 'Unknown')}")
    lines.append(f"Total Funding: {profile.get('total_funding', 'Unknown')}")
    lines.append(f"Latest Round: {profile.get('latest_round', 'Unknown')}")
    lines.append(f"Latest Valuation: {profile.get('latest_valuation', 'Unknown')}")
    lines.append(f"Industry: {profile.get('industry', 'Unknown')}")
    lines.append(f"Key Products: {', '.join(profile.get('key_products', []))}")
    lines.append(f"Target Customers: {profile.get('target_customers', 'Unknown')}")
    return "\n".join(lines)


def _parse_txt_header(raw: str) -> tuple:
    """
    解析 txt 文件头部的 Source / Date / Title 行
    返回 (url, date, title, body)
    """
    lines = raw.strip().split("\n")
    url, date, title = None, None, None
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("source:"):
            url = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("date:"):
            date = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("title:"):
            title = stripped.split(":", 1)[1].strip()
        elif stripped == "":
            body_start = i + 1
            break
        else:
            body_start = i
            break

    body = "\n".join(lines[body_start:]).strip()
    return url, date, title, body


def _classify_source_type(filename: str) -> str:
    if "news" in filename:
        return "news"
    elif "analysis" in filename:
        return "analysis"
    elif "about" in filename:
        return "webpage"
    elif "pricing" in filename:
        return "webpage"
    elif "hiring" in filename:
        return "hiring"
    return "other"


def _parse_pdf(pdf_path: Path, company_name: str) -> Tuple[DocumentMeta, str]:
    """PDF 解析 — MVP 用 pdfplumber 提取纯文本"""
    import pdfplumber

    meta = DocumentMeta(
        source_id=f"{company_name}_{pdf_path.stem}",
        source_type="pdf",
        title=pdf_path.stem,
        company=company_name,
    )

    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

    return meta, "\n\n".join(text_parts)
