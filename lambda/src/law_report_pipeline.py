"""法令レポート生成パイプライン（AWS版）"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from . import prompts
from .bedrock_client import BedrockClient
from .config import AppConfig
from .report_utils import (
    _format_reference,
    _format_reference_for_prompt,
    convert_citation_to_external_link,
    sanitize_mermaid_content,
)
from .retrieval_s3vectors import ArticleWithSummary, FullArticle, S3VectorsRetriever
from .usage_tracker import UsageTracker

logger = logging.getLogger(__name__)

_URL_IN_QUERY_PATTERN = re.compile(r"https://\S+")
_ARTICLE_NUM_PATTERN = re.compile(r"第(\d+)条")
_QUERY_LAW_NAME_PATTERN = re.compile(r"[一-龥ァ-ヴー]+(?:法律|法|規則|政令|条例|省令)")


def _parse_ai_selection(selection_str: str, max_index: int) -> list[int]:
    """AI応答から選択されたインデックスを解析する"""
    selected_indices = []
    try:
        lines = selection_str.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                try:
                    idx = int(line.split(".")[0].split(",")[0].strip())
                    if 1 <= idx <= max_index:
                        selected_indices.append(idx)
                except (ValueError, IndexError):
                    continue
        # カンマ区切りの数字も解析
        if not selected_indices:
            numbers = re.findall(r"\d+", selection_str)
            for num_str in numbers:
                idx = int(num_str)
                if 1 <= idx <= max_index:
                    selected_indices.append(idx)
    except Exception as e:
        logger.error(f"AI選択結果の解析エラー: {e}")

    if not selected_indices:
        logger.warning("AI selection parsing failed. Using default selection strategy.")
        if max_index <= 3:
            return list(range(1, max_index + 1))
        else:
            return [1, max_index // 2, max_index]

    selected_indices = list(set(selected_indices))[:20]
    return selected_indices


def _filter_references_by_citations(report_text: str, all_references: list) -> list:
    """レポート内で実際に引用された参照のみを抽出する"""
    raw_nums = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", report_text)
    cited_indices = sorted(
        {int(n.strip()) for group in raw_nums for n in group.split(",")}
    )
    filtered_refs = [
        (i, all_references[i - 1])
        for i in cited_indices
        if 1 <= i <= len(all_references)
    ]
    return filtered_refs


def _estimate_law_names(
    query: str, bedrock_client: BedrockClient, usage_tracker: UsageTracker
) -> tuple[list[str], list]:
    """法令名を推定して (estimated_law_names, web_hits) を返す"""
    logger.info("Estimating law names with Bedrock web search...")
    estimated_law_names = []

    try:
        today_str = date.today().isoformat()
        search_query = (
            f"以下のクエリに関連する日本の法令名を調査して、JSON形式で回答してください。"
            f"説明文は不要です。JSONのみ出力してください。\n\nクエリ: {query}"
        )

        system_instruction = (
            f"本日の日付は {today_str} です。"
            "クエリに関連する日本の法令を調査し、関連する法令名を以下のJSON形式で回答してください。"
            "調査の際はe-Govや各省庁の公式サイトを優先して参照してください。"
            "必ず有効なJSONのみを出力し、説明文やマークダウンは一切含めないでください："
            '{"law_names": ["法令名1", "法令名2", "法令名3"]}'
            f"\n【重要1】廃止・失効した法令は絶対に含めないこと。本日時点（{today_str}）で"
            "既に廃止・統合されている法令は除外し、現行の後継法令名のみを返すこと。"
            "\n【重要2】クエリで言及された法令名が通称・略称の場合、対応する正式名称が"
            "確実に特定できる場合のみ採用すること。"
        )

        response_text, usage = bedrock_client.generate_text(
            prompt=search_query,
            system_instruction=system_instruction,
            temperature=0.0,
            max_tokens=2048,
            top_p=1.0,
        )
        usage_tracker.add_usage(bedrock_client.config.model_id, usage)

        logger.info(f"Law name estimation response length: {len(response_text)} chars")
        logger.info(f"Law name estimation response preview: {response_text[:500]}...")

        # Stage 1: 直接JSON解析
        stripped_text = re.sub(
            r"^```(?:json)?\s*\n?|```\s*$", "", response_text.strip()
        )
        try:
            result = json.loads(stripped_text)
            estimated_law_names = result.get("law_names", [])
            logger.info(f"Stage 1 success - Direct JSON parsing: {estimated_law_names}")
        except json.JSONDecodeError:
            logger.info("Stage 1 failed - Trying Stage 2: JSON extraction with regex")

            # Stage 2: 正規表現でJSON部分を抽出
            json_pattern = r'\{[^{}]*"law_names"[^{}]*\[[^\]]*\][^{}]*\}'
            json_matches = re.findall(json_pattern, response_text, re.DOTALL)

            for json_match in json_matches:
                try:
                    result = json.loads(json_match)
                    estimated_law_names = result.get("law_names", [])
                    logger.info(
                        f"Stage 2 success - Extracted JSON: {estimated_law_names}"
                    )
                    break
                except json.JSONDecodeError:
                    continue

            # Stage 3: 正規表現で法令名を直接抽出
            if not estimated_law_names:
                logger.info("Stage 2 failed - Trying Stage 3: Direct extraction")
                law_patterns = [
                    r"([^。、\n]*(?:法|規則|省令|政令|条例)[^。、\n]*)",
                ]
                for pattern in law_patterns:
                    matches = re.findall(pattern, response_text)
                    if matches:
                        estimated_law_names = list(
                            {
                                match.strip()
                                for match in matches
                                if 3 < len(match.strip()) < 50
                            }
                        )[:10]
                        logger.info(
                            f"Stage 3 success - Regex extraction: {estimated_law_names}"
                        )
                        break

        if not estimated_law_names:
            logger.warning("All stages failed to extract law names")
        else:
            logger.info(f"Final extracted law names: {estimated_law_names}")

    except Exception as e:
        logger.error(f"Law name estimation failed: {e}")
        return ([], [])

    if not estimated_law_names:
        return ([], [])

    return (estimated_law_names, [])


def _search_articles(
    law_names: list[str], retriever: S3VectorsRetriever
) -> list[ArticleWithSummary]:
    """S3 Vectorsで法令条文を検索する"""
    logger.info("Starting S3 Vectors nearest law articles search...")
    articles = []
    try:
        articles = retriever.get_articles_by_nearest_law(law_names)
        logger.info(f"S3 Vectors found {len(articles)} articles.")

        if not articles:
            logger.warning("No articles found. Trying broader search...")
            broader_terms = [*law_names, "法律", "規則", "政令"]
            articles = retriever.get_articles_by_nearest_law(broader_terms)
            logger.info(f"Broader search found {len(articles)} articles.")
    except Exception as e:
        logger.error(f"S3 Vectors search failed: {e}")
        articles = []

    return articles


def _select_articles(
    query: str,
    articles: list[ArticleWithSummary],
    bedrock_client: BedrockClient,
    usage_tracker: UsageTracker,
) -> list[ArticleWithSummary]:
    """AIで関連条文を選択する"""
    logger.info("AI selecting relevant articles...")
    if len(articles) > 5:
        try:
            summary_list_str = "\n".join(
                [
                    f"{i + 1}. {a.law_title} - {a.article_summary if a.article_summary else '概要なし'}"
                    for i, a in enumerate(articles)
                ]
            )

            prompt = f"元のクエリ: {query}\n\n条文概要リスト:\n{summary_list_str}"
            response_text, usage = bedrock_client.generate_text(
                prompt=prompt,
                system_instruction=prompts.PROMPT_SELECT_RELEVANT_ARTICLES,
                temperature=0.0,
                max_tokens=8192,
                top_p=1.0,
            )
            usage_tracker.add_usage(bedrock_client.config.model_id, usage)

            selected_indices = _parse_ai_selection(response_text, len(articles))
            selected_articles = [
                articles[i - 1]
                for i in selected_indices
                if 1 <= i <= len(articles)
            ]

            if selected_articles:
                articles = selected_articles
                logger.info(f"AI selected {len(articles)} relevant articles")
            else:
                logger.warning("AI article selection failed, using all articles")
        except Exception as e:
            logger.error(f"AI article selection failed: {e}, using all articles")

    return articles


def _to_full_articles(articles: list[ArticleWithSummary]) -> list[FullArticle]:
    """条文データをFullArticle形式に変換する"""
    logger.info("Converting articles to FullArticle format...")
    final_articles = []
    for article in articles:
        if article.content:
            full_article = FullArticle(
                law_id=article.law_id,
                title=article.law_title,
                content=article.content,
                unique_anchor=article.unique_anchor,
                anchor=None,
                url=f"https://laws.e-gov.go.jp/law/{article.law_id.split('_')[0]}",
            )
            final_articles.append(full_article)
    return final_articles


def _build_references(
    final_articles: list[FullArticle], web_hits: list
) -> tuple[list, str]:
    """検索結果をマージして参照リストと参考情報テキストを返す"""
    logger.info("Merging search results...")
    article_search_results = final_articles
    unique_web_search_results = [
        result
        for result in web_hits
        if not any(article.url == result.get("url") for article in final_articles)
    ]

    search_results = [*article_search_results, *unique_web_search_results]

    references_text = "\n\n".join(
        [_format_reference_for_prompt(i, r) for i, r in enumerate(search_results)]
    )

    return (search_results, references_text)


def _expand_law_names_with_ordinances(law_names: list[str]) -> list[str]:
    """法令名リストに施行令・施行規則を補完する"""
    expanded = list(law_names)
    for name in law_names:
        if name.endswith("法律") or name.endswith("法"):
            expanded.append(f"{name}施行令")
            expanded.append(f"{name}施行規則")
    seen: set[str] = set()
    result = []
    for n in expanded:
        if n not in seen:
            seen.add(n)
            result.append(n)
    logger.info(f"Expanded law names: {result}")
    return result


def _bigram_similarity(s1: str, s2: str) -> float:
    """バイグラムJaccard係数で類似度を計算する"""
    _particles = re.compile(r"[をにはがのもとでやへからまで等]")
    s1_norm = _particles.sub("", s1)
    s2_norm = _particles.sub("", s2)

    def bigrams(s: str) -> set:
        return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else set()

    b1 = bigrams(s1_norm)
    b2 = bigrams(s2_norm)
    if not b1 or not b2:
        return 0.0
    return len(b1 & b2) / len(b1 | b2)


def _extract_law_names_from_query(query: str) -> list[str]:
    """クエリから法令名候補を直接抽出する"""
    matches = _QUERY_LAW_NAME_PATTERN.findall(query)
    return [m for m in matches if len(m) >= 4]


def _build_substitution_warning(
    query_law_names: list[str], estimated_law_names: list[str]
) -> str:
    """読み替えが発生した場合の開示指示を生成する"""
    if not query_law_names or not estimated_law_names:
        return ""

    threshold = 0.30
    substituted = []
    for qname in query_law_names:
        sims = {ename: _bigram_similarity(qname, ename) for ename in estimated_law_names}
        best_match = max(sims, key=sims.get)
        best_sim = sims[best_match]
        if best_sim < threshold:
            substituted.append((qname, best_match))

    if not substituted:
        return ""

    lines = ["【読み替え通知 - 回答の冒頭で必ず開示すること】"]
    for original, replacement in substituted:
        lines.append(
            f"ユーザーが指定した法令名「{original}」は実在しない可能性があります。"
            f"最も近い実在法令「{replacement}」として回答しますが、"
            f"「{original}」が通称・略称、または実在しない法令名である可能性を"
            f"回答の冒頭で明示してください。"
        )
    lines += ["---", ""]
    return "\n".join(lines)


def _check_law_name_divergence(
    law_names: list[str], articles: list[ArticleWithSummary]
) -> str:
    """推定法令名とBQ取得法令名の乖離を検出する"""
    if not law_names or not articles:
        return ""

    bq_law_titles = list(
        {a.law_title for a in articles if getattr(a, "law_title", None)}
    )
    if not bq_law_titles:
        return ""

    threshold = 0.40
    diverged = []
    for law_name in law_names:
        sims = {title: _bigram_similarity(law_name, title) for title in bq_law_titles}
        best_title = max(sims, key=sims.get)
        best_sim = sims[best_title]
        if best_sim < threshold:
            diverged.append((law_name, best_title, best_sim))

    if not diverged:
        return ""

    lines = [
        "【警告】推定された法令名と取得された法令名に大きな乖離があります。",
        "",
    ]
    for estimated, actual, sim in diverged:
        lines.append(
            f"- 推定法令名 「{estimated}」 → 取得法令 「{actual}」（類似度: {sim:.0%}）"
        )
    lines += ["", "---", ""]
    return "\n".join(lines)


def _build_mentioned_articles_prefix(
    query: str, articles: list[ArticleWithSummary]
) -> str:
    """クエリで言及された条文番号に対応する条文を抽出しプレフィックスを生成する"""
    mentioned_nums = _ARTICLE_NUM_PATTERN.findall(query)
    if not mentioned_nums:
        return ""

    matched = []
    for num in mentioned_nums:
        pattern = re.compile(rf"Article_{num}$")
        for article in articles:
            if hasattr(article, "unique_anchor") and pattern.search(
                article.unique_anchor
            ):
                matched.append((num, article))
                break

    if not matched:
        return ""

    lines = [
        "【クエリで指定された条文の照合情報 - 回答前に必ず確認すること】",
        "",
    ]
    for num, article in matched:
        summary = getattr(article, "article_summary", None) or ""
        lines.append(f"■ 第{num}条の正式タイトル: {summary}")

    lines += ["", "---", ""]
    return "\n".join(lines)


def _generate_complete_report(
    query: str,
    references_text: str,
    bedrock_client: BedrockClient,
    usage_tracker: UsageTracker,
) -> str:
    """1回でレポート全体を生成する"""
    logger.info("Generating complete report in single generation...")

    prompt = f"クエリ: {query}\n\n参考情報:\n{references_text}"
    response_text, usage = bedrock_client.generate_text(
        prompt=prompt,
        system_instruction=prompts.PROMPT_GENERATE_COMPLETE_REPORT,
        temperature=0.0,
        max_tokens=8192,
        top_p=1.0,
    )
    usage_tracker.add_usage(bedrock_client.config.model_id, usage)

    # モデルが # 見出し前に出力する前置き文を除去
    first_heading = re.search(r"^#", response_text, re.MULTILINE)
    if first_heading:
        response_text = response_text[first_heading.start() :]

    return response_text


def _finalize_report(report_text: str, search_results: list) -> str:
    """引用リンク変換・Mermaid安全化・参照セクション結合を行い最終レポートを返す"""
    logger.info("Filtering references based on citations...")
    filtered_references = _filter_references_by_citations(report_text, search_results)

    if not filtered_references:
        logger.warning("No citation markers found, using all references as fallback")
        filtered_references = [(i + 1, ref) for i, ref in enumerate(search_results)]

    if filtered_references and isinstance(filtered_references[0], tuple):
        filtered_references_text = "\n\n".join(
            [
                _format_reference(original_idx - 1, ref)
                for original_idx, ref in filtered_references
            ]
        )
    else:
        filtered_references_text = "\n\n".join(
            [_format_reference(i, r) for i, r in enumerate(filtered_references)]
        )

    temp_final_report_with_links = convert_citation_to_external_link(
        report_text, filtered_references
    )
    temp_final_report_sanitized = sanitize_mermaid_content(temp_final_report_with_links)

    final_report = "\n\n".join(
        [temp_final_report_sanitized, "## 出典", filtered_references_text]
    )

    actual_ref_count = len(filtered_references) if filtered_references else 0
    logger.info(f"Report completed. Using {actual_ref_count} filtered references.")

    return final_report


def generate_law_report(
    query: str,
    bedrock_client: BedrockClient,
    config: AppConfig,
    retriever: S3VectorsRetriever,
) -> tuple[str, list[dict]]:
    """法令レポート生成のメイン関数"""
    usage_tracker = UsageTracker()

    # クエリから元の法令名を抽出
    query_law_names = _extract_law_names_from_query(query)

    # 法令名推定
    law_names, web_hits = _estimate_law_names(query, bedrock_client, usage_tracker)
    if not law_names:
        return (
            "クエリから関連する法令を特定できませんでした。より具体的な法令名を含めてクエリを再構成してください。",
            [],
        )

    # 施行令・施行規則を補完
    search_law_names = _expand_law_names_with_ordinances(law_names)

    # S3 Vectors 検索
    articles = _search_articles(search_law_names, retriever)

    if not articles:
        return (
            "申し訳ございませんが、該当する法令が見つかりませんでした。",
            [],
        )

    # 読み替えチェック
    substitution_warning = _build_substitution_warning(query_law_names, law_names)
    law_name_divergence_warning = _check_law_name_divergence(law_names, articles)

    # AI による条文選択
    articles = _select_articles(query, articles, bedrock_client, usage_tracker)

    # 条文番号プレフィックス
    mentioned_prefix = _build_mentioned_articles_prefix(query, articles)

    # FullArticle形式に変換
    final_articles = _to_full_articles(articles)
    if not final_articles:
        return "該当する条文が見つかりませんでした。", []

    # 参照情報構築
    search_results, references_text = _build_references(final_articles, web_hits)

    # 警告・照合情報を参考情報の先頭に埋め込む
    if mentioned_prefix:
        references_text = mentioned_prefix + references_text
    if law_name_divergence_warning:
        references_text = law_name_divergence_warning + references_text
    if substitution_warning:
        references_text = substitution_warning + references_text

    # レポート生成
    report = _generate_complete_report(
        query, references_text, bedrock_client, usage_tracker
    )

    # 最終レポート構築
    final_report = _finalize_report(report, search_results)

    usage_summary = usage_tracker.get_usage_summary()
    logger.info(f"Usage summary: {usage_summary}")
    return final_report, usage_summary
