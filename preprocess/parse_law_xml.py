"""法令XMLパーサー

e-Gov法令XMLを解析し、条文ごとのJSONLデータを生成する。
Google Cloud版 load_to_bq.py のXMLパース部分を移植。
"""

import json
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def get_raw_text(element):
    """XML要素からテキストを取得する"""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def format_article_text(article_element):
    """条文要素をフォーマットされたテキストに変換する"""
    if article_element is None:
        return ""

    lines = []

    def get_full_text(element):
        if element is None:
            return ""
        return "".join(element.itertext()).strip()

    indent_map = {
        "Article": 0,
        "Paragraph": 1,
        "Item": 2,
        "Subitem1": 3,
        "Subitem2": 4,
        "Subitem3": 5,
        "Subitem4": 6,
        "Subitem5": 7,
        "Subitem6": 8,
        "Subitem7": 9,
        "Subitem8": 10,
        "Subitem9": 11,
        "Subitem10": 12,
        "List": 2,
        "Table": 2,
    }

    title_tags = [f"Subitem{i}Title" for i in range(1, 11)] + [
        "ArticleCaption",
        "ArticleTitle",
        "ParagraphNum",
        "ItemTitle",
    ]
    sentence_tags = [f"Subitem{i}Sentence" for i in range(1, 11)] + [
        "ParagraphSentence",
        "ItemSentence",
    ]

    def recursive_format(element, level):
        is_structural_node = element.tag in indent_map
        if is_structural_node:
            parts = []
            title_elements = [el for el in element if el.tag in title_tags]
            sentence_elements = [el for el in element if el.tag in sentence_tags]

            parts.extend(get_full_text(el) for el in title_elements)
            parts.extend(get_full_text(el) for el in sentence_elements)

            if parts:
                lines.append("\u3000" * level + "\u3000".join(parts))

        child_level = level + 1 if is_structural_node else level

        for child in element:
            if child.tag in title_tags or child.tag in sentence_tags:
                continue
            if child.tag not in indent_map:
                child_text = get_full_text(child)
                if child_text:
                    lines.append("\u3000" * child_level + child_text)
            else:
                recursive_format(child, child_level)

    recursive_format(article_element, 0)
    return "\n".join(lines)


def parse_law_xml(xml_file):
    """法令XMLファイルを解析して条文チャンクのリストを返す"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    law_title = get_raw_text(root.find(".//LawTitle"))
    law_num = get_raw_text(root.find(".//LawNum"))

    chunks = []

    def process_article(article, provision_prefix):
        if article.get("Delete") == "true":
            return

        article_num = article.get("Num")
        unique_anchor = f"{provision_prefix}_Article_{article_num}"

        egov_anchor = None
        if provision_prefix == "Main":
            egov_anchor = f"Mp-At_{article_num.replace('_', '_')}"

        content = format_article_text(article)
        article_caption = get_raw_text(article.find("ArticleCaption"))
        first_paragraph = article.find(".//Paragraph")
        first_paragraph_text = (
            get_raw_text(first_paragraph.find(".//ParagraphSentence"))
            if first_paragraph is not None
            else ""
        )
        article_summary = article_caption or first_paragraph_text

        chunks.append(
            {
                "law_num": law_num,
                "law_title": law_title,
                "unique_anchor": unique_anchor,
                "anchor": egov_anchor,
                "content": content,
                "article_summary": article_summary,
            }
        )

    # 本則の条文
    for article in root.findall(".//MainProvision//Article"):
        process_article(article, "Main")

    # 附則の条文（改正附則を除く）
    for suppl_provision in root.findall(".//SupplProvision"):
        if "AmendLawNum" in suppl_provision.attrib:
            continue
        for article in suppl_provision.findall(".//Article"):
            process_article(article, "Suppl")

    return chunks


def process_file(file_path):
    """1つのXMLファイルを処理してJSONL行のリストを返す"""
    try:
        law_id = os.path.splitext(os.path.basename(file_path))[0]
        tree = ET.parse(file_path)
        xml_root = tree.getroot()

        era = xml_root.get("Era")
        year_str = xml_root.get("Year")
        year = int(year_str) if year_str and year_str.isdigit() else 0

        law_type = xml_root.get("LawType")
        promulgate_month_str = xml_root.get("PromulgateMonth")
        promulgate_day_str = xml_root.get("PromulgateDay")
        promulgate_month = (
            int(promulgate_month_str)
            if promulgate_month_str and promulgate_month_str.isdigit()
            else 1
        )
        promulgate_day = (
            int(promulgate_day_str)
            if promulgate_day_str and promulgate_day_str.isdigit()
            else 1
        )

        if era == "Meiji":
            gregorian_year = 1867 + year
        elif era == "Taisho":
            gregorian_year = 1911 + year
        elif era == "Showa":
            gregorian_year = 1925 + year
        elif era == "Heisei":
            gregorian_year = 1988 + year
        elif era == "Reiwa":
            gregorian_year = 2018 + year
        else:
            gregorian_year = year

        promulgate_date = (
            f"{gregorian_year:04d}-{promulgate_month:02d}-{promulgate_day:02d}"
        )

        article_chunks = parse_law_xml(file_path)

        rows = [
            {
                "law_id": law_id,
                "law_num": chunk["law_num"],
                "law_title": chunk["law_title"],
                "unique_anchor": chunk["unique_anchor"],
                "anchor": chunk["anchor"],
                "content": chunk["content"],
                "article_summary": chunk["article_summary"],
                "era": era,
                "year": year,
                "law_type": law_type,
                "promulgate_date": promulgate_date,
            }
            for chunk in article_chunks
        ]
        return rows, file_path
    except Exception as e:
        print(f"ERROR: Error processing file {file_path}: {e}", file=sys.stderr)
        return [], file_path


def parse_all_xml_files(source_directory: str, output_file: str) -> int:
    """全XMLファイルを解析してJSONLファイルに出力する

    Returns:
        処理された行数
    """
    files_to_process = []
    for root_dir, _, files in os.walk(source_directory):
        for file in files:
            if file.endswith(".xml"):
                files_to_process.append(os.path.join(root_dir, file))

    if not files_to_process:
        print("INFO: No XML files found.", file=sys.stderr)
        return 0

    total_rows = 0
    with ProcessPoolExecutor() as executor, open(
        output_file, "w", encoding="utf-8"
    ) as f:
        future_to_file = {
            executor.submit(process_file, file_path): file_path
            for file_path in files_to_process
        }

        for future in tqdm(
            as_completed(future_to_file),
            total=len(files_to_process),
            desc="Processing XML files",
        ):
            rows, _ = future.result()
            if rows:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += 1

    print(
        f"INFO: Processed {len(files_to_process)} files, {total_rows} rows written to {output_file}",
        file=sys.stderr,
    )
    return total_rows


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python parse_law_xml.py <source_directory> <output_jsonl_file>",
            file=sys.stderr,
        )
        sys.exit(1)

    source_dir = sys.argv[1]
    output_file = sys.argv[2]
    count = parse_all_xml_files(source_dir, output_file)
    print(f"Total rows: {count}")
