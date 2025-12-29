#!/usr/bin/env python3
"""
BI-RADS HTML Exporter
=====================
데이터베이스의 BI-RADS 가이드라인을 HTML로 추출
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import sqlite3
import re
from datetime import datetime

def create_anchor(text):
    """텍스트를 URL-safe 앵커로 변환"""
    # 소문자 변환 및 특수문자 제거
    anchor = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
    anchor = re.sub(r'\s+', '-', anchor)
    return anchor

def export_birads_to_html():
    """BI-RADS 가이드라인을 HTML로 추출"""

    db_path = Path('data/index/metadata.db')
    output_path = Path('src/ui/static/birads_guidelines.html')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 데이터베이스 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 모든 BI-RADS 문서 가져오기
    cursor.execute('''
        SELECT pmid, title, full_content, authors, journal, year
        FROM papers
        WHERE pmid LIKE 'BIRADS_%'
        ORDER BY title
    ''')

    documents = cursor.fetchall()
    conn.close()

    # Assessment Categories 먼저 정렬
    categories = {
        'Category 0': [],
        'Category 1': [],
        'Category 2': [],
        'Category 3': [],
        'Category 4': [],
        'Category 4A': [],
        'Category 4B': [],
        'Category 4C': [],
        'Category 5': [],
        'Category 6': [],
        'Other': []
    }

    for doc in documents:
        pmid, title, content, authors, journal, year = doc

        # 카테고리별로 분류
        if 'Category 0' in title:
            categories['Category 0'].append(doc)
        elif 'Category 1' in title:
            categories['Category 1'].append(doc)
        elif 'Category 2' in title:
            categories['Category 2'].append(doc)
        elif 'Category 3' in title or 'category 3' in title:
            categories['Category 3'].append(doc)
        elif 'Category 4C' in title or '4C' in title:
            categories['Category 4C'].append(doc)
        elif 'Category 4B' in title or '4B' in title:
            categories['Category 4B'].append(doc)
        elif 'Category 4A' in title or '4A' in title:
            categories['Category 4A'].append(doc)
        elif 'Category 4' in title:
            categories['Category 4'].append(doc)
        elif 'Category 5' in title:
            categories['Category 5'].append(doc)
        elif 'Category 6' in title:
            categories['Category 6'].append(doc)
        else:
            categories['Other'].append(doc)

    # HTML 생성
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BI-RADS 2025 Guidelines</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        header p {{
            font-size: 1.1em;
            opacity: 0.9;
        }}

        nav {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            position: sticky;
            top: 20px;
            z-index: 100;
        }}

        nav h2 {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #667eea;
        }}

        nav ul {{
            list-style: none;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
        }}

        nav a {{
            display: block;
            padding: 10px 15px;
            background: #f8f9fa;
            color: #333;
            text-decoration: none;
            border-radius: 5px;
            transition: all 0.3s;
            border-left: 3px solid #667eea;
        }}

        nav a:hover {{
            background: #667eea;
            color: white;
            transform: translateX(5px);
        }}

        .category-section {{
            background: white;
            margin: 30px 0;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}

        .category-section h2 {{
            font-size: 2em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .document {{
            margin: 30px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #764ba2;
        }}

        .document h3 {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 15px;
        }}

        .document-meta {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 15px;
            padding: 10px;
            background: white;
            border-radius: 4px;
        }}

        .document-content {{
            font-size: 1.05em;
            line-height: 1.8;
            white-space: pre-wrap;
            background: white;
            padding: 20px;
            border-radius: 5px;
        }}

        .back-to-top {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #667eea;
            color: white;
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            text-decoration: none;
            font-size: 1.5em;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
            transition: all 0.3s;
        }}

        .back-to-top:hover {{
            background: #764ba2;
            transform: translateY(-5px);
        }}

        footer {{
            text-align: center;
            padding: 40px 20px;
            color: #666;
            margin-top: 50px;
        }}

        @media (max-width: 768px) {{
            header h1 {{
                font-size: 1.8em;
            }}

            nav ul {{
                grid-template-columns: 1fr;
            }}

            .category-section {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <header>
        <h1>📘 BI-RADS 2025 Guidelines</h1>
        <p>Breast Imaging Reporting and Data System</p>
        <p style="font-size: 0.9em; margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </header>

    <div class="container">
        <nav id="top">
            <h2>📑 목차 (Table of Contents)</h2>
            <ul>
"""

    # 목차 생성
    for cat_name, docs in categories.items():
        if docs:
            anchor = create_anchor(cat_name)
            count = len(docs)
            html += f'                <li><a href="#{anchor}">{cat_name} ({count}개 문서)</a></li>\n'

    html += """            </ul>
        </nav>

"""

    # 각 카테고리별 문서 추가
    for cat_name, docs in categories.items():
        if not docs:
            continue

        cat_anchor = create_anchor(cat_name)
        html += f"""        <section class="category-section" id="{cat_anchor}">
            <h2>{cat_name}</h2>
"""

        for pmid, title, content, authors, journal, year in docs:
            doc_anchor = create_anchor(pmid)

            html += f"""            <article class="document" id="{doc_anchor}">
                <h3>{title}</h3>
                <div class="document-meta">
                    <strong>저자:</strong> {authors or 'ACR Committee on BI-RADS'}<br>
                    <strong>출처:</strong> {journal or 'ACR BI-RADS Atlas'} ({year or '2025'})<br>
                    <strong>문서 ID:</strong> {pmid}
                </div>
                <div class="document-content">{content or '내용 없음'}</div>
            </article>
"""

        html += "        </section>\n\n"

    # Footer
    html += f"""    </div>

    <a href="#top" class="back-to-top" title="맨 위로">↑</a>

    <footer>
        <p>Generated by Sophia AI - MARIA-Mammo</p>
        <p>Total Documents: {len(documents)}</p>
        <p style="font-size: 0.9em; color: #999; margin-top: 10px;">
            이 문서는 교육 및 연구 목적으로 제공되며, 의학적 조언을 대체할 수 없습니다.
        </p>
    </footer>
</body>
</html>
"""

    # HTML 파일 저장
    output_path.write_text(html, encoding='utf-8')

    print(f"✅ BI-RADS HTML 생성 완료!")
    print(f"📁 파일: {output_path}")
    print(f"📊 총 {len(documents)}개 문서 포함")
    print(f"\n카테고리별 문서 수:")
    for cat_name, docs in categories.items():
        if docs:
            print(f"   {cat_name}: {len(docs)}개")

if __name__ == "__main__":
    export_birads_to_html()
