# marker-pdf Integration Research — bbox + page boundaries

Research date: 2026-07-18. Investigated whether marker-pdf can provide
block-level bounding boxes and page numbers for chunk-level citation.

## TL;DR

Both remaining structural debt items (`bbox` always NULL, `page_start`/`page_end`
always NULL) are **wireable now**. marker-pdf already produces this data — lean's
current `MarkdownRenderer` path strips it. Switching to `JSONRenderer` (or using
both) would give us everything needed.

## Evidence

### marker's JSON output format

From `/datalab-to/marker` master branch (`_autodocs/types.md`):

```python
class JSONBlockOutput(BaseModel):
    id: str
    block_type: str
    html: str
    polygon: List[List[float]]       # 4 corners [[x,y], [x,y], [x,y], [x,y]]
    bbox: List[float]                # [x0, y0, x1, y1] computed from polygon
    children: List["JSONBlockOutput"] | None
    section_hierarchy: Dict[int, str] | None
    images: dict | None
```

### marker's flat/chunked output

```python
class FlatBlockOutput(BaseModel):
    id: str
    block_type: str
    html: str
    page: int                         # PAGE NUMBER — exactly what page_start/page_end needs
    polygon: List[List[float]]
    bbox: List[float]
    section_hierarchy: Dict[int, str] | None
    images: dict | None
```

### PolygonBox geometry

From `marker/schema/polygon.py`:

```python
class PolygonBox(BaseModel):
    polygon: List[List[float]]

    @property
    def bbox(self) -> List[float]:
        # Computed: [min_x, min_y, max_x, max_y]
```

Every block in marker's `Document` model has a `polygon: PolygonBox` field.

## Implementation path

### Local path (`_extract_local` in `marker_converter.py`)

Current:
```python
config = {"output_format": "markdown", "force_ocr": force_ocr}
# ...
text, _, images = text_from_rendered(rendered)
```

The rendered output is a `MarkdownOutput` — markdown text + images only. The
`Document` object inside the converter has full structure (blocks with polygons,
page numbers), but it's discarded after rendering.

Options:
1. **Use `JSONRenderer` instead of `MarkdownRenderer`** — produces
   `JSONBlockOutput` tree with bbox + polygon per block. Tradeoff: lose the
   clean markdown text; would need to reconstruct or run both renderers.
2. **Access the `Document` object directly** — after `converter.__call__(pdf)`,
   the converter has `self.page_count` and the internal `Document` has full
   structure. But the rendered output is the public API; reaching inside the
   converter is fragile.
3. **Use `ChunkRenderer`** — produces `FlatBlockOutput` with `page: int` +
   `bbox: List[float]`. Purpose-built for RAG chunking. Closest to lean's use
   case.

Recommended: option 3 (`ChunkRenderer`) — it's designed for exactly this use
case (RAG with page + bbox metadata). It produces flat blocks with page numbers
and bounding boxes, ready for chunking.

### Remote path (`_extract_remote` + `scripts/marker_server.py`)

Current `marker_server.py` returns:
```json
{"markdown": "...", "page_count": N, "images": {...}}
```

Would need to return:
```json
{
  "markdown": "...",
  "page_count": N,
  "images": {...},
  "blocks": [
    {"id": "...", "block_type": "...", "html": "...", "page": 0,
     "bbox": [x0, y0, x1, y1], "polygon": [[x,y], ...]}
  ]
}
```

The server already has the `PdfConverter` in GPU memory — switching the renderer
to `ChunkRenderer` or adding a second render pass is straightforward.

### Chunker changes

`ChunkResult` currently carries: `section_path`, `heading_text`, `chunk_index`,
`token_count`, `content`. Would need to add:
- `bbox: list[float] | None` — from the first/last block in the chunk
- `page_start: int | None` — from the first block's page
- `page_end: int | None` — from the last block's page

`build_sections` would need to thread bbox + page from marker's block output
into each `Section`. `chunk_sections` would propagate them to `ChunkResult`.

### Store changes

`ChunkRow` already has `bbox`, `page_start`, `page_end` fields — they're just
always `None` because nothing populates them. No schema change needed.

## Effort estimate

| Step | Effort | Risk |
|---|---|---|
| Switch local renderer to `ChunkRenderer`, extract structured blocks | 2-3 hrs | Low — additive, existing tests cover markdown path |
| Update remote `marker_server.py` to return blocks + bbox | 1-2 hrs | Low — additive JSON fields |
| Thread bbox + page through chunker (`Section` → `ChunkResult`) | 3-4 hrs | Medium — touches chunker internals |
| Update `_build_chunk_rows` to populate from `ChunkResult` | 1 hr | Low |
| Tests for the new data path | 2-3 hrs | — |
| **Total** | **~1-1.5 days** | — |

## Dependencies

- marker-pdf is already installed (`uv sync --extra marker`)
- No new pip dependencies needed
- `ChunkRenderer` exists in marker master; verify it's in the installed version
- DB schema already has `bbox`, `page_start`, `page_end` columns (migration 003 + 011)

## Next steps

1. Verify `ChunkRenderer` is available in the installed marker version
2. Prototype: render one PDF with `ChunkRenderer`, inspect output
3. If output matches `FlatBlockOutput` schema → implement the wiring
4. Update `docs/limitations.md` to remove the "always NULL" caveats

This research unblocks AGENTS.md debt items #1 (page_start/page_end) and #2 (bbox).
