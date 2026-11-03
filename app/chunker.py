import hashlib
import re
from uuid import UUID

from app.models import Chunk, Page
from app.text_cleaner import clean_text


def generate_chunk_id(
    tenant_id: UUID,
    document_id: str,
    chunk_index: int,
    chunk_text: str,
) -> str:
    """Generate a stable, tenant-specific chunk ID."""

    identity = (
        f"{tenant_id}:{document_id}:{chunk_index}:{chunk_text}"
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()


def _validate_chunk_settings(
    chunk_size: int,
    overlap: int,
) -> None:
    """Validate settings shared by all chunking strategies."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _validate_pages(pages: list[Page]) -> None:
    """Ensure all pages belong to the same document and filename."""

    if not pages:
        return

    document_id = pages[0].document_id
    filename = pages[0].filename

    if any(
        page.document_id != document_id or page.filename != filename
        for page in pages
    ):
        raise ValueError("All pages must belong to the same document")


def chunk_pages(
    pages: list[Page],
    chunk_size: int = 400,
    overlap: int = 60,
    *,
    tenant_id: UUID,
) -> list[Chunk]:
    """
    Split pages into overlapping word-based chunks.

    Preserve paragraph boundaries and track the source pages
    represented in every chunk.
    """

    _validate_chunk_settings(chunk_size, overlap)

    if not pages:
        return []

    _validate_pages(pages)

    document_id = pages[0].document_id
    filename = pages[0].filename

    words: list[str] = []
    word_pages: list[int] = []
    paragraph_starts: list[bool] = []

    for page in pages:
        cleaned_text = clean_text(page.text)

        for paragraph in cleaned_text.split("\n\n"):
            paragraph_words = paragraph.split()

            for word_index, word in enumerate(paragraph_words):
                words.append(word)
                word_pages.append(page.page_number)
                paragraph_starts.append(word_index == 0)

    if not words:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))

        chunk_text = words[start]

        for global_index in range(start + 1, end):
            separator = (
                "\n\n"
                if paragraph_starts[global_index]
                else " "
            )
            chunk_text += separator + words[global_index]

        source_pages = list(dict.fromkeys(word_pages[start:end]))
        chunk_index = len(chunks)

        chunks.append(
            Chunk(
                chunk_id=generate_chunk_id(
                    tenant_id=tenant_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    chunk_text=chunk_text,
                ),
                tenant_id=tenant_id,
                document_id=document_id,
                filename=filename,
                chunk_index=chunk_index,
                text=chunk_text,
                start_page=source_pages[0],
                end_page=source_pages[-1],
                source_pages=source_pages,
            )
        )

        if end == len(words):
            break

    return chunks


def chunk_pages_independently(
    pages: list[Page],
    chunk_size: int = 400,
    overlap: int = 60,
    *,
    tenant_id: UUID,
) -> list[Chunk]:
    """
    Experimental strategy: chunk each page independently.

    Chunks never combine text from different PDF pages.
    """

    _validate_chunk_settings(chunk_size, overlap)

    if not pages:
        return []

    _validate_pages(pages)

    chunks: list[Chunk] = []
    step = chunk_size - overlap

    for page in pages:
        words = clean_text(page.text).split()

        if not words:
            continue

        for start in range(0, len(words), step):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_index = len(chunks)

            chunks.append(
                Chunk(
                    chunk_id=generate_chunk_id(
                        tenant_id=tenant_id,
                        document_id=page.document_id,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text,
                    ),
                    tenant_id=tenant_id,
                    document_id=page.document_id,
                    filename=page.filename,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_page=page.page_number,
                    end_page=page.page_number,
                    source_pages=[page.page_number],
                )
            )

            if end == len(words):
                break

    return chunks


def extract_abstract_chunk(
    pages: list[Page],
    *,
    tenant_id: UUID,
) -> Chunk | None:
    """
    Extract the abstract as a standalone chunk.

    Stops at the next numbered section heading.
    """

    abstract_heading = re.compile(
        r"^\s*(?:\d+\s+)?abstract\s*$",
        re.IGNORECASE,
    )

    section_heading = re.compile(
        r"^\s*\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z /&()\-,:]*\s*$"
    )

    for page in pages:
        lines = page.text.splitlines()

        for index, line in enumerate(lines):
            if not abstract_heading.match(line):
                continue

            abstract_lines = []

            for next_line in lines[index + 1:]:
                if section_heading.match(next_line):
                    break

                abstract_lines.append(next_line)

            abstract_text = clean_text("\n".join(abstract_lines))

            if not abstract_text:
                return None

            return Chunk(
                chunk_id=generate_chunk_id(
                    tenant_id=tenant_id,
                    document_id=page.document_id,
                    chunk_index=page.page_number,
                    chunk_text=abstract_text,
                ),
                tenant_id=tenant_id,
                document_id=page.document_id,
                filename=page.filename,
                chunk_index=page.page_number,
                text=abstract_text,
                start_page=page.page_number,
                end_page=page.page_number,
                source_pages=[page.page_number],
            )

    return None


def chunk_pages_by_section(
    pages: list[Page],
    chunk_size: int = 400,
    overlap: int = 60,
    *,
    tenant_id: UUID,
) -> list[Chunk]:
    """
    Create section-aware chunks without changing the production chunker.

    Guarantees:
    - Skips the TOC when its start and the Abstract are identified.
    - Keeps title-page text separate from the Abstract.
    - Recognizes numbered section headings, not numbered list items.
    - Preserves the source page of every word across page boundaries.
    - Keeps each chunk within chunk_size words, including its heading.
    - Applies overlap within each section, never across sections.
    """

    _validate_chunk_settings(chunk_size, overlap)

    if not pages:
        return []

    _validate_pages(pages)

    document_id = pages[0].document_id
    filename = pages[0].filename

    # ---------------------------------------------------------
    # 1. Find the Table of Contents and the Abstract.
    # ---------------------------------------------------------

    toc_start_index = None
    abstract_page_index = None

    for page_index, page in enumerate(pages):
        lines = page.text.splitlines()

        has_toc_heading = any(
            re.fullmatch(
                r"\s*table\s+of\s+contents\s*",
                line,
                flags=re.IGNORECASE,
            )
            for line in lines
        )

        if toc_start_index is None and has_toc_heading:
            toc_start_index = page_index
            continue

        if toc_start_index is not None and page_index > toc_start_index:
            has_abstract_heading = any(
                re.fullmatch(
                    r"\s*abstract\s*",
                    line,
                    flags=re.IGNORECASE,
                )
                for line in lines
            )

            if has_abstract_heading:
                abstract_page_index = page_index
                break

    # Only skip the TOC if the beginning of the Abstract is found.
    if (
        toc_start_index is not None
        and abstract_page_index is not None
    ):
        pages_to_process = (
            pages[:toc_start_index]
            + pages[abstract_page_index:]
        )
    else:
        pages_to_process = pages

    # ---------------------------------------------------------
    # 2. Define heading patterns.
    # ---------------------------------------------------------

    heading_pattern = re.compile(
        r"^\s*(?:\d+\s+|\d+(?:\.\d+)+\s+)"
        r"[A-Z][A-Za-z0-9][A-Za-z0-9 /&()'’.,:–—\-]*\s*$"
    )

    abstract_pattern = re.compile(
        r"^\s*abstract\s*$",
        re.IGNORECASE,
    )

    # ---------------------------------------------------------
    # 3. Build sections with word-to-page mapping.
    # ---------------------------------------------------------

    sections = []

    current_section = {
        "heading": "",
        "heading_page": pages_to_process[0].page_number,
        "paragraphs": [],
    }

    paragraph_lines: list[tuple[str, int]] = []

    def finish_paragraph():
        """Save paragraph text and the page number of each word."""

        nonlocal paragraph_lines

        if not paragraph_lines:
            return

        paragraph_text = " ".join(
            line for line, _ in paragraph_lines
        ).strip()

        paragraph_word_pages = []

        for line, page_number in paragraph_lines:
            paragraph_word_pages.extend(
                [page_number] * len(line.split())
            )

        if paragraph_text:
            current_section["paragraphs"].append(
                (paragraph_text, paragraph_word_pages)
            )

        paragraph_lines = []

    def finish_section():
        """Finish and save the current section."""

        finish_paragraph()

        if current_section["heading"] or current_section["paragraphs"]:
            sections.append(current_section)

    for page in pages_to_process:
        cleaned_page_text = clean_text(page.text)

        for line in cleaned_page_text.splitlines():
            stripped_line = line.strip()

            if not stripped_line:
                finish_paragraph()
                continue

            # Ignore standalone printed page numbers.
            if re.fullmatch(r"\d+", stripped_line):
                continue

            if abstract_pattern.fullmatch(stripped_line):
                finish_section()

                current_section = {
                    "heading": "Abstract",
                    "heading_page": page.page_number,
                    "paragraphs": [],
                }
                continue

            if heading_pattern.fullmatch(stripped_line):
                finish_section()

                current_section = {
                    "heading": stripped_line,
                    "heading_page": page.page_number,
                    "paragraphs": [],
                }
                continue

            paragraph_lines.append(
                (stripped_line, page.page_number)
            )

    finish_section()

    # ---------------------------------------------------------
    # 4. Convert sections into chunks.
    # ---------------------------------------------------------

    chunks: list[Chunk] = []

    for section in sections:
        heading = section["heading"]
        heading_page = section["heading_page"]
        heading_words = heading.split() if heading else []

        if len(heading_words) >= chunk_size:
            raise ValueError(
                f"Section heading has {len(heading_words)} words, "
                f"which does not fit within chunk_size={chunk_size}: "
                f"{heading!r}"
            )

        words: list[str] = []
        word_pages: list[int] = []
        paragraph_starts: list[bool] = []

        for paragraph_text, paragraph_word_pages in section["paragraphs"]:
            paragraph_words = paragraph_text.split()

            if len(paragraph_words) != len(paragraph_word_pages):
                raise ValueError(
                    "Word/page mapping mismatch in section "
                    f"{heading!r}: {len(paragraph_words)} words but "
                    f"{len(paragraph_word_pages)} page mappings."
                )

            for word_index, word in enumerate(paragraph_words):
                words.append(word)
                word_pages.append(paragraph_word_pages[word_index])
                paragraph_starts.append(word_index == 0)

        # Keep heading-only sections as valid chunks.
        if not words:
            if heading:
                chunk_index = len(chunks)

                chunks.append(
                    Chunk(
                        chunk_id=generate_chunk_id(
                            tenant_id=tenant_id,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            chunk_text=heading,
                        ),
                        tenant_id=tenant_id,
                        document_id=document_id,
                        filename=filename,
                        chunk_index=chunk_index,
                        text=heading,
                        start_page=heading_page,
                        end_page=heading_page,
                        source_pages=[heading_page],
                    )
                )

            continue

        body_limit = chunk_size - len(heading_words)

        # Always advance by at least one word.
        step = body_limit - min(overlap, body_limit - 1)

        for start in range(0, len(words), step):
            end = min(start + body_limit, len(words))
            body_text = ""

            for local_index, word in enumerate(words[start:end]):
                global_index = start + local_index

                if local_index == 0:
                    body_text = word
                elif paragraph_starts[global_index]:
                    body_text += "\n\n" + word
                else:
                    body_text += " " + word

            chunk_text = (
                f"{heading}\n{body_text}"
                if heading
                else body_text
            )

            source_pages = []

            # The heading is included in every chunk's text, but its
            # source page is added only to the first chunk in a section
            # to avoid misleading page metadata for later chunks.
            if heading and start == 0:
                source_pages.append(heading_page)

            for page_number in word_pages[start:end]:
                if page_number not in source_pages:
                    source_pages.append(page_number)

            chunk_index = len(chunks)

            chunks.append(
                Chunk(
                    chunk_id=generate_chunk_id(
                        tenant_id=tenant_id,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        chunk_text=chunk_text,
                    ),
                    tenant_id=tenant_id,
                    document_id=document_id,
                    filename=filename,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    start_page=min(source_pages),
                    end_page=max(source_pages),
                    source_pages=source_pages,
                )
            )

            if end == len(words):
                break

    return chunks