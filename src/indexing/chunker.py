"""
BI-RADS Document Chunking Utility
==================================
Lexicon 문서를 의미 있는 청크로 분할하여 검색 정확도 향상
"""

import re
from typing import List, Dict
from src.models import Paper


class BiRadsChunker:
    """BI-RADS 문서 청크 생성기"""

    def __init__(self, max_chunk_size: int = 1000):
        """
        Args:
            max_chunk_size: 최대 청크 크기 (문자 수)
        """
        self.max_chunk_size = max_chunk_size

    def chunk_section_ii(self, paper: Paper) -> List[Dict[str, str]]:
        """
        SECTION_II (Lexicon) 테이블을 항목별로 청크 분할

        Args:
            paper: SECTION_II 논문 객체

        Returns:
            청크 리스트. 각 청크는 {"pmid": str, "content": str, "title": str} 형태
        """
        if not paper.full_content:
            return []

        chunks = []
        content = paper.full_content

        # 각 Finding 카테고리별로 분할
        # A. Masses, B. Calcifications, C. Architectural Distortion 등

        # 1. Masses 섹션 추출
        mass_pattern = r'\*\*A\. Masses\*\*.*?(?=\*\*[B-Z]\.|\Z)'
        mass_match = re.search(mass_pattern, content, re.DOTALL)
        if mass_match:
            mass_content = mass_match.group(0)

            # Margin, Shape 등을 개별 청크로
            # Margin 청크
            if "Margin" in mass_content or "margin" in mass_content:
                margin_chunk = self._extract_table_row(mass_content, "Margin|margin")
                if margin_chunk:
                    chunks.append({
                        "pmid": f"{paper.pmid}_CHUNK_MASS_MARGIN",
                        "content": f"# BI-RADS Mammography Lexicon - Mass Margin\n\n{margin_chunk}",
                        "title": "BI-RADS Mass Margin Classification",
                    })

            # Shape 청크
            if "Shape" in mass_content or "shape" in mass_content:
                shape_chunk = self._extract_table_row(mass_content, "Shape|shape")
                if shape_chunk:
                    chunks.append({
                        "pmid": f"{paper.pmid}_CHUNK_MASS_SHAPE",
                        "content": f"# BI-RADS Mammography Lexicon - Mass Shape\n\n{shape_chunk}",
                        "title": "BI-RADS Mass Shape Classification",
                    })

            # Density 청크
            if "Density" in mass_content or "density" in mass_content:
                density_chunk = self._extract_table_row(mass_content, "Density|density")
                if density_chunk:
                    chunks.append({
                        "pmid": f"{paper.pmid}_CHUNK_MASS_DENSITY",
                        "content": f"# BI-RADS Mammography Lexicon - Mass Density\n\n{density_chunk}",
                        "title": "BI-RADS Mass Density Classification",
                    })

        # 2. Calcifications 섹션 추출
        calc_pattern = r'\*\*B\. Calcifications\*\*.*?(?=\*\*[C-Z]\.|\Z)'
        calc_match = re.search(calc_pattern, content, re.DOTALL)
        if calc_match:
            calc_content = calc_match.group(0)

            # Morphology 청크
            if "Morphology" in calc_content or "morphology" in calc_content:
                morph_chunk = self._extract_table_row(calc_content, "Morphology|morphology")
                if morph_chunk:
                    chunks.append({
                        "pmid": f"{paper.pmid}_CHUNK_CALC_MORPHOLOGY",
                        "content": f"# BI-RADS Mammography Lexicon - Calcification Morphology\n\n{morph_chunk}",
                        "title": "BI-RADS Calcification Morphology Classification",
                    })

            # Distribution 청크
            if "Distribution" in calc_content or "distribution" in calc_content:
                dist_chunk = self._extract_table_row(calc_content, "Distribution|distribution")
                if dist_chunk:
                    chunks.append({
                        "pmid": f"{paper.pmid}_CHUNK_CALC_DISTRIBUTION",
                        "content": f"# BI-RADS Mammography Lexicon - Calcification Distribution\n\n{dist_chunk}",
                        "title": "BI-RADS Calcification Distribution Classification",
                    })

        # 3. Architectural Distortion 청크
        arch_pattern = r'\*\*C\. Architectural [Dd]istortion\*\*.*?(?=\*\*[D-Z]\.|\Z)'
        arch_match = re.search(arch_pattern, content, re.DOTALL)
        if arch_match:
            arch_content = arch_match.group(0)
            chunks.append({
                "pmid": f"{paper.pmid}_CHUNK_ARCH_DISTORTION",
                "content": f"# BI-RADS Mammography Lexicon - Architectural Distortion\n\n{arch_content}",
                "title": "BI-RADS Architectural Distortion",
            })

        # 4. Asymmetries 청크
        asym_pattern = r'\*\*D\. Asymmetr.*?(?=\*\*[E-Z]\.|\Z)'
        asym_match = re.search(asym_pattern, content, re.DOTALL)
        if asym_match:
            asym_content = asym_match.group(0)
            chunks.append({
                "pmid": f"{paper.pmid}_CHUNK_ASYMMETRIES",
                "content": f"# BI-RADS Mammography Lexicon - Asymmetries\n\n{asym_content}",
                "title": "BI-RADS Asymmetries Classification",
            })

        # 원본 전체 문서도 유지 (fallback)
        chunks.append({
            "pmid": paper.pmid,
            "content": content,
            "title": paper.title or "BI-RADS Mammography Lexicon",
        })

        return chunks

    def _extract_table_row(self, content: str, row_pattern: str) -> str:
        """
        테이블에서 특정 row 추출

        Args:
            content: 테이블 내용
            row_pattern: 찾을 row 패턴 (정규식)

        Returns:
            추출된 row 내용
        """
        # Markdown 테이블 row 찾기
        # 예: | **A. Masses** | *Margin* | Circumscribed, Obscured, Indistinct, Spiculated |

        # 먼저 해당 키워드가 포함된 줄 찾기
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if re.search(row_pattern, line, re.IGNORECASE):
                # 테이블 row인 경우
                if '|' in line:
                    # 다음 몇 줄도 같이 포함 (multi-line 테이블 row)
                    extracted = [line]
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if lines[j].strip().startswith('|'):
                            extracted.append(lines[j])
                        elif lines[j].strip() and not lines[j].strip().startswith('|'):
                            break
                    return '\n'.join(extracted)
                else:
                    # 일반 텍스트인 경우 해당 단락 추출
                    return self._extract_paragraph(lines, i)

        return ""

    def _extract_paragraph(self, lines: List[str], start_idx: int) -> str:
        """
        특정 줄부터 단락 추출

        Args:
            lines: 전체 줄 리스트
            start_idx: 시작 줄 인덱스

        Returns:
            추출된 단락
        """
        paragraph = []

        # 이전 줄들 중 관련된 내용 찾기
        for i in range(max(0, start_idx - 3), start_idx):
            if lines[i].strip():
                paragraph.append(lines[i])

        # 현재 줄
        paragraph.append(lines[start_idx])

        # 다음 줄들 추가
        for i in range(start_idx + 1, min(start_idx + 10, len(lines))):
            line = lines[i].strip()
            if not line:  # 빈 줄이면 단락 끝
                break
            if line.startswith('#'):  # 새로운 헤더면 끝
                break
            paragraph.append(lines[i])

        return '\n'.join(paragraph)

    def should_chunk(self, paper: Paper) -> bool:
        """
        해당 문서를 청킹해야 하는지 판단

        Args:
            paper: 논문 객체

        Returns:
            청킹 필요 여부
        """
        # SECTION_IV (Findings) 섹션들을 청킹
        return paper.pmid.startswith("BIRADS_2025_SECTION_IV")

    def chunk_paper(self, paper: Paper) -> List[Dict[str, str]]:
        """
        논문을 청크로 분할

        Args:
            paper: 논문 객체

        Returns:
            청크 리스트
        """
        if not self.should_chunk(paper):
            return []

        # SECTION_IV_A: Masses
        if paper.pmid == "BIRADS_2025_SECTION_IV_A":
            return self.chunk_masses(paper)

        # SECTION_IV_B: Calcifications
        elif paper.pmid.startswith("BIRADS_2025_SECTION_IV_B"):
            return self.chunk_calcifications(paper)

        # SECTION_IV_C: Architectural Distortion
        elif paper.pmid == "BIRADS_2025_SECTION_IV_C":
            return self.chunk_simple(paper, "Architectural Distortion")

        # SECTION_IV_D: Asymmetries
        elif paper.pmid == "BIRADS_2025_SECTION_IV_D":
            return self.chunk_asymmetries(paper)

        return []

    def chunk_masses(self, paper: Paper) -> List[Dict[str, str]]:
        """
        Masses 섹션을 Shape, Margin, Density로 청크 분할

        Args:
            paper: SECTION_IV_A 논문 객체

        Returns:
            청크 리스트
        """
        if not paper.full_content:
            return []

        chunks = []
        content = paper.full_content

        # Shape 청크
        shape_pattern = r'(?:###?\s*1\.\s*SHAPE|###?\s*Shape).*?(?=###?\s*[2-9]\.|\Z)'
        shape_match = re.search(shape_pattern, content, re.DOTALL | re.IGNORECASE)
        if shape_match:
            chunks.append({
                "pmid": f"{paper.pmid}_CHUNK_SHAPE",
                "content": f"# BI-RADS Mammography - Mass Shape\n\n{shape_match.group(0)}",
                "title": "BI-RADS Mass Shape Classification",
            })

        # Margin 청크
        margin_pattern = r'(?:###?\s*2\.\s*MARGIN|###?\s*Margin).*?(?=###?\s*[3-9]\.|\Z)'
        margin_match = re.search(margin_pattern, content, re.DOTALL | re.IGNORECASE)
        if margin_match:
            chunks.append({
                "pmid": f"{paper.pmid}_CHUNK_MARGIN",
                "content": f"# BI-RADS Mammography - Mass Margin\n\n{margin_match.group(0)}",
                "title": "BI-RADS Mass Margin Classification",
            })

        # Density 청크
        density_pattern = r'(?:###?\s*3\.\s*DENSITY|###?\s*Density).*?(?=###?\s*[4-9]\.|\Z)'
        density_match = re.search(density_pattern, content, re.DOTALL | re.IGNORECASE)
        if density_match:
            chunks.append({
                "pmid": f"{paper.pmid}_CHUNK_DENSITY",
                "content": f"# BI-RADS Mammography - Mass Density\n\n{density_match.group(0)}",
                "title": "BI-RADS Mass Density Classification",
            })

        # 원본 전체 문서도 유지
        chunks.append({
            "pmid": paper.pmid,
            "content": content,
            "title": paper.title or "BI-RADS Masses",
        })

        return chunks

    def chunk_calcifications(self, paper: Paper) -> List[Dict[str, str]]:
        """
        Calcifications 섹션을 각 타입별로 청크 분할

        Args:
            paper: SECTION_IV_B* 논문 객체

        Returns:
            청크 리스트
        """
        if not paper.full_content:
            return []

        chunks = []
        content = paper.full_content

        # SECTION_IV_B1: Typically Benign - 각 타입별로 분할
        if paper.pmid == "BIRADS_2025_SECTION_IV_B1":
            # 각 calcification 타입 (Skin, Vascular, Coarse, etc.)을 개별 청크로
            patterns = [
                ("SKIN", r'(?:####?\s*a\.\s*Skin).*?(?=####?\s*[b-z]\.|\Z)'),
                ("VASCULAR", r'(?:####?\s*b\.\s*Vascular).*?(?=####?\s*[c-z]\.|\Z)'),
                ("COARSE", r'(?:####?\s*c\.\s*Coarse).*?(?=####?\s*[d-z]\.|\Z)'),
                ("LARGE_ROD", r'(?:####?\s*d\.\s*Large rod-like).*?(?=####?\s*[e-z]\.|\Z)'),
                ("ROUND", r'(?:####?\s*e\.\s*Round).*?(?=####?\s*[f-z]\.|\Z)'),
                ("RIM", r'(?:####?\s*f\.\s*Rim).*?(?=####?\s*[g-z]\.|\Z)'),
                ("LAYERING", r'(?:####?\s*g\.\s*Layering).*?(?=####?\s*[h-z]\.|\Z)'),
                ("SUTURE", r'(?:####?\s*h\.\s*Suture).*?(?=####?\s*[i-z]\.|\Z)'),
            ]

            for chunk_name, pattern in patterns:
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    chunks.append({
                        "pmid": f"{paper.pmid}_CHUNK_{chunk_name}",
                        "content": f"# BI-RADS Calcifications - {chunk_name.replace('_', ' ').title()}\n\n{match.group(0)}",
                        "title": f"BI-RADS Calcification: {chunk_name.replace('_', ' ').title()}",
                    })

        # SECTION_IV_B2: Suspicious Morphology
        elif paper.pmid == "BIRADS_2025_SECTION_IV_B2":
            patterns = [
                ("AMORPHOUS", r'(?:####?\s*a\.\s*Amorphous).*?(?=####?\s*[b-z]\.|\Z)'),
                ("COARSE_HETEROGENEOUS", r'(?:####?\s*b\.\s*Coarse heterogeneous).*?(?=####?\s*[c-z]\.|\Z)'),
                ("FINE_PLEOMORPHIC", r'(?:####?\s*c\.\s*Fine pleomorphic).*?(?=####?\s*[d-z]\.|\Z)'),
                ("FINE_LINEAR", r'(?:####?\s*d\.\s*Fine linear).*?(?=####?\s*[e-z]\.|\Z)'),
            ]

            for chunk_name, pattern in patterns:
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    chunks.append({
                        "pmid": f"{paper.pmid}_CHUNK_{chunk_name}",
                        "content": f"# BI-RADS Calcifications - {chunk_name.replace('_', ' ').title()}\n\n{match.group(0)}",
                        "title": f"BI-RADS Calcification: {chunk_name.replace('_', ' ').title()}",
                    })

        # SECTION_IV_B3: Distribution
        elif paper.pmid == "BIRADS_2025_SECTION_IV_B3":
            patterns = [
                ("DIFFUSE", r'(?:####?\s*a\.\s*Diffuse).*?(?=####?\s*[b-z]\.|\Z)'),
                ("REGIONAL", r'(?:####?\s*b\.\s*Regional).*?(?=####?\s*[c-z]\.|\Z)'),
                ("GROUPED", r'(?:####?\s*c\.\s*Grouped).*?(?=####?\s*[d-z]\.|\Z)'),
                ("LINEAR", r'(?:####?\s*d\.\s*Linear).*?(?=####?\s*[e-z]\.|\Z)'),
                ("SEGMENTAL", r'(?:####?\s*e\.\s*Segmental).*?(?=####?\s*[f-z]\.|\Z)'),
            ]

            for chunk_name, pattern in patterns:
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                if match:
                    chunks.append({
                        "pmid": f"{paper.pmid}_CHUNK_{chunk_name}",
                        "content": f"# BI-RADS Calcifications - {chunk_name.replace('_', ' ').title()}\n\n{match.group(0)}",
                        "title": f"BI-RADS Calcification Distribution: {chunk_name.replace('_', ' ').title()}",
                    })

        # 원본 전체 문서도 유지
        chunks.append({
            "pmid": paper.pmid,
            "content": content,
            "title": paper.title or "BI-RADS Calcifications",
        })

        return chunks

    def chunk_asymmetries(self, paper: Paper) -> List[Dict[str, str]]:
        """
        Asymmetries 섹션을 각 타입별로 청크 분할

        Args:
            paper: SECTION_IV_D 논문 객체

        Returns:
            청크 리스트
        """
        if not paper.full_content:
            return []

        chunks = []
        content = paper.full_content

        # 각 asymmetry 타입별로 분할
        patterns = [
            ("GLOBAL_ASYMMETRY", r'(?:###?\s*1\.\s*GLOBAL ASYMMETRY|###?\s*Global asymmetry).*?(?=###?\s*[2-9]\.|\Z)'),
            ("ASYMMETRY", r'(?:###?\s*2\.\s*ASYMMETRY|###?\s*[^G]\s*Asymmetry).*?(?=###?\s*[3-9]\.|\Z)'),
            ("FOCAL_ASYMMETRY", r'(?:###?\s*3\.\s*FOCAL ASYMMETRY|###?\s*Focal asymmetry).*?(?=###?\s*[4-9]\.|\Z)'),
        ]

        for chunk_name, pattern in patterns:
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                chunks.append({
                    "pmid": f"{paper.pmid}_CHUNK_{chunk_name}",
                    "content": f"# BI-RADS Asymmetries - {chunk_name.replace('_', ' ').title()}\n\n{match.group(0)}",
                    "title": f"BI-RADS {chunk_name.replace('_', ' ').title()}",
                })

        # 원본 전체 문서도 유지
        chunks.append({
            "pmid": paper.pmid,
            "content": content,
            "title": paper.title or "BI-RADS Asymmetries",
        })

        return chunks

    def chunk_simple(self, paper: Paper, section_name: str) -> List[Dict[str, str]]:
        """
        간단한 섹션은 원본만 유지 (청킹하지 않음)

        Args:
            paper: 논문 객체
            section_name: 섹션 이름

        Returns:
            원본 문서를 포함한 리스트
        """
        return [{
            "pmid": paper.pmid,
            "content": paper.full_content or "",
            "title": paper.title or f"BI-RADS {section_name}",
        }]
