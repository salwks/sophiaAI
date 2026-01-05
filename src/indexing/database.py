"""
Sophia AI: Database Manager
=============================
SQLite (메타데이터) + LanceDB (벡터) 통합 관리
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lancedb
import numpy as np
import pyarrow as pa

from src.models import Paper, QueryFilters

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite + LanceDB 통합 데이터베이스 매니저"""

    def __init__(self, db_path: Path = Path("data/index")):
        """
        Args:
            db_path: 데이터베이스 디렉토리
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.sqlite_path = self.db_path / "metadata.db"
        self.lance_path = self.db_path / "lancedb"  # Changed from "vectors" to "lancedb"

        self._init_sqlite()
        self._init_lancedb()

        logger.info(f"Database initialized at {self.db_path}")

    def _init_sqlite(self):
        """SQLite 초기화"""
        self.sqlite_conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self.sqlite_conn.row_factory = sqlite3.Row

        cursor = self.sqlite_conn.cursor()

        # papers 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS papers (
                pmid TEXT PRIMARY KEY,
                doi TEXT,
                title TEXT NOT NULL,
                authors TEXT,
                journal TEXT,
                journal_abbrev TEXT,
                year INTEGER,
                month INTEGER,
                abstract TEXT,
                mesh_terms TEXT,
                keywords TEXT,
                publication_types TEXT,
                modality TEXT,
                pathology TEXT,
                study_type TEXT,
                population TEXT,
                citation_count INTEGER DEFAULT 0,
                journal_if REAL,
                pmc_id TEXT,
                full_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        # full_content 컬럼 추가 (기존 테이블의 경우)
        try:
            cursor.execute("ALTER TABLE papers ADD COLUMN full_content TEXT")
            self.sqlite_conn.commit()
            logger.info("Added full_content column to papers table")
        except sqlite3.OperationalError:
            # 이미 컬럼이 존재하는 경우
            pass

        # 인덱스 생성
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_journal ON papers(journal)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_doi ON papers(doi)")

        self.sqlite_conn.commit()
        logger.info(f"SQLite initialized: {self.sqlite_path}")

    def _init_lancedb(self):
        """LanceDB 초기화"""
        self.lance_db = lancedb.connect(str(self.lance_path))
        self.lance_table = None

        # 테이블 존재 확인
        try:
            self.lance_table = self.lance_db.open_table("papers")
            logger.info(f"LanceDB table opened: {len(self.lance_table)} vectors")
        except FileNotFoundError:
            logger.info("LanceDB table not found, will create on first insert")
        except Exception as e:
            logger.error(f"Error opening LanceDB table: {e}", exc_info=True)
            logger.info("Will attempt to open table on first search")

    def insert_paper(self, paper: Paper, embedding: Optional[np.ndarray] = None):
        """단일 논문 삽입"""
        self._insert_paper_sqlite(paper)

        if embedding is not None:
            self._insert_embedding_lance(paper.pmid, embedding)

    def insert_papers(
        self,
        papers: List[Paper],
        embeddings: Optional[List[np.ndarray]] = None,
    ):
        """배치 논문 삽입"""
        # SQLite 배치 삽입
        cursor = self.sqlite_conn.cursor()

        for paper in papers:
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO papers (
                        pmid, doi, title, authors, journal, journal_abbrev,
                        year, month, abstract, mesh_terms, keywords,
                        publication_types, modality, pathology, study_type,
                        population, citation_count, journal_if, pmc_id, full_content,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper.pmid,
                        paper.doi,
                        paper.title,
                        json.dumps(paper.authors),
                        paper.journal,
                        paper.journal_abbrev,
                        paper.year,
                        paper.month,
                        paper.abstract,
                        json.dumps(paper.mesh_terms),
                        json.dumps(paper.keywords),
                        json.dumps(paper.publication_types),
                        json.dumps(paper.modality),
                        json.dumps(paper.pathology),
                        paper.study_type,
                        paper.population,
                        paper.citation_count,
                        paper.journal_if,
                        paper.pmc_id,
                        getattr(paper, 'full_content', None),
                        datetime.now().isoformat(),
                        None,
                    ),
                )
            except Exception as e:
                logger.warning(f"Failed to insert paper {paper.pmid}: {e}")

        self.sqlite_conn.commit()
        logger.info(f"Inserted {len(papers)} papers to SQLite")

        # LanceDB 배치 삽입
        if embeddings:
            self._insert_embeddings_lance(
                [p.pmid for p in papers],
                embeddings,
            )

    def _insert_paper_sqlite(self, paper: Paper):
        """SQLite에 단일 논문 삽입"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO papers (
                pmid, doi, title, authors, journal, journal_abbrev,
                year, month, abstract, mesh_terms, keywords,
                publication_types, modality, pathology, study_type,
                population, citation_count, journal_if, pmc_id, full_content, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper.pmid,
                paper.doi,
                paper.title,
                json.dumps(paper.authors),
                paper.journal,
                paper.journal_abbrev,
                paper.year,
                paper.month,
                paper.abstract,
                json.dumps(paper.mesh_terms),
                json.dumps(paper.keywords),
                json.dumps(paper.publication_types),
                json.dumps(paper.modality),
                json.dumps(paper.pathology),
                paper.study_type,
                paper.population,
                paper.citation_count,
                paper.journal_if,
                paper.pmc_id,
                getattr(paper, 'full_content', None),
                datetime.now().isoformat(),
            ),
        )
        self.sqlite_conn.commit()

    def _insert_embedding_lance(self, pmid: str, embedding: np.ndarray):
        """LanceDB에 단일 임베딩 삽입"""
        self._insert_embeddings_lance([pmid], [embedding])

    def _insert_embeddings_lance(
        self,
        pmids: List[str],
        embeddings: List[np.ndarray],
    ):
        """LanceDB에 배치 임베딩 삽입"""
        if not pmids or not embeddings:
            return

        data = [
            {"pmid": pmid, "vector": emb.tolist()}
            for pmid, emb in zip(pmids, embeddings)
        ]

        if self.lance_table is None:
            # 테이블 생성
            self.lance_table = self.lance_db.create_table("papers", data)
            logger.info(f"Created LanceDB table with {len(data)} vectors")
        else:
            # 기존 테이블에 추가
            self.lance_table.add(data)
            logger.info(f"Added {len(data)} vectors to LanceDB")

    def search_metadata(self, filters: QueryFilters) -> List[str]:
        """
        메타데이터 기반 필터 검색

        Returns:
            필터에 맞는 PMID 리스트
        """
        cursor = self.sqlite_conn.cursor()

        conditions = []
        params = []

        if filters.year_min:
            conditions.append("year >= ?")
            params.append(filters.year_min)

        if filters.year_max:
            conditions.append("year <= ?")
            params.append(filters.year_max)

        if filters.modality:
            modality_conditions = []
            for m in filters.modality:
                modality_conditions.append("modality LIKE ?")
                params.append(f'%"{m}"%')
            conditions.append(f"({' OR '.join(modality_conditions)})")

        if filters.pathology:
            pathology_conditions = []
            for p in filters.pathology:
                pathology_conditions.append("pathology LIKE ?")
                params.append(f'%"{p}"%')
            conditions.append(f"({' OR '.join(pathology_conditions)})")

        if filters.study_type:
            conditions.append("study_type = ?")
            params.append(filters.study_type)

        if filters.population:
            conditions.append("population = ?")
            params.append(filters.population)

        if filters.min_citations:
            conditions.append("citation_count >= ?")
            params.append(filters.min_citations)

        query = "SELECT pmid FROM papers"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor.execute(query, params)
        results = cursor.fetchall()

        return [row["pmid"] for row in results]

    def search_vector(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
        pmid_filter: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        벡터 검색

        Returns:
            (pmid, score) 튜플 리스트
        """
        # Lazy initialization - try to open table if not already opened
        if self.lance_table is None:
            try:
                self.lance_table = self.lance_db.open_table("papers")
                logger.info(f"LanceDB table lazy-loaded: {len(self.lance_table)} vectors")
            except Exception as e:
                logger.warning(f"LanceDB table not initialized: {e}")
                return []

        try:
            results = self.lance_table.search(query_embedding.tolist()).limit(k * 2).to_list()

            # 필터 적용
            if pmid_filter:
                pmid_set = set(pmid_filter)
                results = [r for r in results if r["pmid"] in pmid_set]

            # 상위 k개
            results = results[:k]

            return [(r["pmid"], 1 - r["_distance"]) for r in results]

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def get_paper(self, pmid: str) -> Optional[Paper]:
        """PMID로 논문 조회"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT * FROM papers WHERE pmid = ?", (pmid,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_paper(row)

    def get_papers(self, pmids: List[str]) -> List[Paper]:
        """여러 PMID로 논문 조회"""
        if not pmids:
            return []

        cursor = self.sqlite_conn.cursor()
        placeholders = ",".join("?" * len(pmids))
        cursor.execute(f"SELECT * FROM papers WHERE pmid IN ({placeholders})", pmids)
        rows = cursor.fetchall()

        # 순서 유지
        papers_dict = {self._row_to_paper(row).pmid: self._row_to_paper(row) for row in rows}
        return [papers_dict[pmid] for pmid in pmids if pmid in papers_dict]

    def _safe_json_loads(self, value: str, default: list = None) -> list:
        """안전한 JSON 파싱 (에러 처리 포함)"""
        if default is None:
            default = []

        if not value:
            return default

        # 공백 제거
        value = value.strip()

        if not value:
            return default

        try:
            result = json.loads(value)
            return result if isinstance(result, list) else default
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"JSON parse error for value '{value[:50]}...': {e}, using default {default}")
            return default

    def _row_to_paper(self, row: sqlite3.Row) -> Paper:
        """SQLite Row를 Paper 객체로 변환"""
        # full_content 필드가 있는지 확인
        try:
            full_content = row["full_content"]
        except (KeyError, IndexError):
            full_content = None

        return Paper(
            pmid=row["pmid"],
            doi=row["doi"],
            title=row["title"],
            authors=self._safe_json_loads(row["authors"]),
            journal=row["journal"] or "",
            journal_abbrev=row["journal_abbrev"] or "",
            year=row["year"] or 0,
            month=row["month"],
            abstract=row["abstract"] or "",
            mesh_terms=self._safe_json_loads(row["mesh_terms"]),
            keywords=self._safe_json_loads(row["keywords"]),
            publication_types=self._safe_json_loads(row["publication_types"]),
            modality=self._safe_json_loads(row["modality"]),
            pathology=self._safe_json_loads(row["pathology"]),
            study_type=row["study_type"],
            population=row["population"],
            citation_count=row["citation_count"] or 0,
            journal_if=row["journal_if"],
            pmc_id=row["pmc_id"],
            full_content=full_content,
        )

    def get_all_pmids(self) -> List[str]:
        """모든 PMID 조회"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("SELECT pmid FROM papers")
        return [row["pmid"] for row in cursor.fetchall()]

    def get_stats(self) -> Dict:
        """데이터베이스 통계"""
        cursor = self.sqlite_conn.cursor()

        # 총 논문 수
        cursor.execute("SELECT COUNT(*) as count FROM papers")
        total = cursor.fetchone()["count"]

        # 연도별
        cursor.execute("SELECT year, COUNT(*) as count FROM papers GROUP BY year ORDER BY year")
        by_year = {row["year"]: row["count"] for row in cursor.fetchall()}

        # 벡터 수
        vector_count = len(self.lance_table) if self.lance_table else 0

        return {
            "total_papers": total,
            "total_vectors": vector_count,
            "by_year": by_year,
        }

    def clear(self):
        """모든 데이터 삭제"""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("DELETE FROM papers")
        self.sqlite_conn.commit()

        if self.lance_table:
            self.lance_db.drop_table("papers")
            self.lance_table = None

        logger.info("Database cleared")

    def close(self):
        """연결 종료"""
        self.sqlite_conn.close()
