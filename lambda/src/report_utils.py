"""レポート生成ユーティリティ"""

import logging
import re

from .retrieval_s3vectors import FullArticle

logger = logging.getLogger(__name__)


def _build_ref_map(references: list) -> dict:
    """references リストを {citation_num: ref_data} の辞書に変換する"""
    ref_map = {}
    for i, item in enumerate(references, start=1):
        if isinstance(item, tuple):
            original_index, ref_data = item
            ref_map[original_index] = ref_data
        else:
            ref_map[i] = item
    return ref_map


def _normalize_content(text: str, max_len: int = 200) -> str:
    """content/snippet をインライン表示用に正規化する"""
    normalized = re.sub(r"\n\u3000*", " ", text).strip()
    return normalized[:max_len]


def _format_reference(i, r):
    """参照情報をフォーマットする"""
    if isinstance(r, FullArticle):
        title = r.title
        content = _normalize_content(r.content) if r.content else ""
        url = str(r.url) if r.url else ""
        content_line = f"\n　　> {content}..." if content else ""
        if url:
            return f"[{i + 1}] 🔗 **[{title}]({url})**{content_line}"
        else:
            return f"[{i + 1}] **{title}**{content_line}"
    elif hasattr(r, "title"):
        title = r.title
        content = _normalize_content(r.content) if r.content else ""
        url = str(r.url) if r.url else ""
        content_line = f"\n　　> {content}..." if content else ""
        if url:
            return f"[{i + 1}] 🔗 **[{title}]({url})**{content_line}"
        else:
            return f"[{i + 1}] **{title}**{content_line}"
    else:
        title = r.get("title", "No title")
        content = _normalize_content(r.get("snippet", ""))
        url = r.get("url", "")
        content_line = f"\n　　> {content}" if content else ""
        if url:
            return f"[{i + 1}] 🔗 **[{title}]({url})**{content_line}"
        else:
            return f"[{i + 1}] **{title}**{content_line}"


def _format_reference_for_prompt(i, r):
    """Bedrockプロンプト向けフォーマット（URLなし・e-laws条文はラベル付き全文）"""
    if isinstance(r, FullArticle) or hasattr(r, "title"):
        content = r.content if hasattr(r, "content") else ""
        title = r.title if hasattr(r, "title") else ""
        return f"[{i + 1}] 【e-laws公式条文】 {title}\n{content}"
    else:
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        content_line = f"\n{snippet}" if snippet else ""
        return f"[{i + 1}] {title}{content_line}"


def sanitize_mermaid_content(text):
    """Mermaidコードブロック内の危険な記号を安全な文字に置換する"""

    def sanitize_mermaid_block(match):
        mermaid_content = match.group(1)

        protected_patterns = {
            "___ARROW_RIGHT___": "-->",
            "___ARROW_LEFT___": "<--",
            "___ARROW_BOTH___": "<-->",
            "___ARROW_DOTTED___": "-.-",
            "___ARROW_THICK___": "===",
            "___ARROW_OPEN___": "---",
            "___COLON_SPACE___": ": ",
            "___PIPE_PIPE___": "||",
            "___AMP_AMP___": "&&",
        }

        protected_content = mermaid_content
        for placeholder, pattern in protected_patterns.items():
            protected_content = protected_content.replace(pattern, placeholder)

        def sanitize_label_content(label_content):
            label_replacements = [
                ("(", "（"),
                (")", "）"),
                ("[", "［"),
                ("]", "］"),
                ("{", "｛"),
                ("}", "｝"),
                ("・", "/"),
                ("#", "＃"),
                ("*", "＊"),
                ('"', "\u201c"),
                ("'", "\u2018"),
                ("<", "＜"),
                (">", "＞"),
                ("&", "＆"),
                ("\n", "<br>"),
            ]
            sanitized_label = label_content
            for old_char, new_char in label_replacements:
                sanitized_label = sanitized_label.replace(old_char, new_char)
            return sanitized_label

        sanitized = protected_content
        sanitized = re.sub(
            r"\(([^)]+)\)",
            lambda m: f"({sanitize_label_content(m.group(1))})",
            sanitized,
        )
        sanitized = re.sub(
            r"\[([^\]]+)\]",
            lambda m: f"[{sanitize_label_content(m.group(1))}]",
            sanitized,
        )
        sanitized = re.sub(
            r"\{([^}]+)\}",
            lambda m: f"{{{sanitize_label_content(m.group(1))}}}",
            sanitized,
        )

        for placeholder, pattern in protected_patterns.items():
            sanitized = sanitized.replace(placeholder, pattern)

        lines = sanitized.split("\n")
        normalized_lines = []
        for line in lines:
            if "-->" in line or "<--" in line or "---" in line or "===" in line:
                normalized_lines.append(line)
            else:
                normalized_lines.append(re.sub(r"\s+", " ", line.strip()))

        return f"```mermaid\n{chr(10).join(normalized_lines)}\n```"

    return re.sub(
        r"```mermaid\n(.*?)\n```", sanitize_mermaid_block, text, flags=re.DOTALL
    )


def convert_citation_to_external_link(text, references):
    """本文中の[数字]表記を対応する参照の外部URLにリンク化する"""
    ref_map = _build_ref_map(references)

    def _link_single(num: int) -> str:
        ref_data = ref_map.get(num)
        if ref_data is None:
            return f"[{num}]"
        if isinstance(ref_data, FullArticle):
            return f"[[{num}]]({ref_data.url})"
        elif hasattr(ref_data, "url") and ref_data.url:
            return f"[[{num}]]({ref_data.url})"
        elif isinstance(ref_data, dict) and ref_data.get("url"):
            return f"[[{num}]]({ref_data['url']})"
        return f"[{num}]"

    def replace_citation(match):
        inner = match.group(1)
        nums = [int(s.strip()) for s in inner.split(",")]
        if len(nums) == 1:
            return _link_single(nums[0])
        return " ".join(_link_single(n) for n in nums)

    mermaid_blocks = [
        (match.start(), match.end())
        for match in re.finditer(r"```mermaid\n(.*?)\n```", text, re.DOTALL)
    ]

    def is_in_mermaid_block(pos):
        return any(start <= pos <= end for start, end in mermaid_blocks)

    def conditional_replace(match):
        if is_in_mermaid_block(match.start()):
            return match.group(0)
        else:
            return replace_citation(match)

    return re.sub(r"\[(\d+(?:,\s*\d+)*)\]", conditional_replace, text)
