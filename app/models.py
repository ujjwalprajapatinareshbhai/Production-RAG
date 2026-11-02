from dataclasses import dataclass

@dataclass
class Page:
    document_id: str
    filename: str
    page_number: int
    text: str

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    chunk_index: int
    text: str
    start_page: int
    end_page: int
    source_pages: list[int]