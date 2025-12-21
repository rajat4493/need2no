from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
from PIL import Image, ImageOps


@dataclass
class PageImage:
    index: int
    path: Path
    width: int
    height: int
    scale: float
    render_dpi: int
    source: str


def prepare_input_images(input_path: Path, outdir: Path, dpi: int = 350) -> Tuple[List[PageImage], Dict[str, object]]:
    input_path = input_path.expanduser().resolve()
    pages_dir = outdir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    page_images: List[PageImage] = []
    trace: Dict[str, object] = {"input": {"path": str(input_path), "render_dpi": dpi}}

    if input_path.suffix.lower() == ".pdf":
        doc = fitz.open(str(input_path))
        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        try:
            for idx, page in enumerate(doc):
                pix = page.get_pixmap(matrix=matrix)
                out_path = pages_dir / f"{input_path.stem}_page_{idx + 1}.png"
                pix.save(str(out_path))
                page_images.append(
                    PageImage(
                        index=idx,
                        path=out_path,
                        width=pix.width,
                        height=pix.height,
                        scale=scale,
                        render_dpi=dpi,
                        source="pdf",
                    )
                )
        finally:
            doc.close()
    else:
        image = Image.open(str(input_path))
        image = ImageOps.exif_transpose(image)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        width, height = image.size
        out_path = pages_dir / f"{input_path.stem}_page_1.png"
        image.save(out_path)
        page_images.append(
            PageImage(
                index=0,
                path=out_path,
                width=width,
                height=height,
                scale=1.0,
                render_dpi=dpi,
                source="image",
            )
        )

    trace["input"].update({"page_count": len(page_images)})
    trace["pages"] = [
        {"index": page.index, "width": page.width, "height": page.height, "scale": page.scale, "source": page.source}
        for page in page_images
    ]
    return page_images, trace


__all__ = ["PageImage", "prepare_input_images"]
