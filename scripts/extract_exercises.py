#!/usr/bin/env python3
import os
import re
import json
import fitz            # PyMuPDF
import pytesseract
import numpy as np
import cv2
from PIL import Image
from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn

# ===========================
# CONFIGURATION
# ===========================

DPI = 300
OCR_LANG = "eng"

# Script root (…/scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Project root (one level above scripts/)
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

PDF_PATH = os.path.join(ROOT, "data", "java.pdf")
OUT_IMAGES = os.path.join(ROOT, "site", "data", "images")
OUT_INDEX = os.path.join(ROOT, "site", "data", "index.json")

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(os.path.dirname(OUT_INDEX), exist_ok=True)

print("PDF PATH:", PDF_PATH)
print("IMAGE OUTPUT:", OUT_IMAGES)
print("INDEX OUTPUT:", OUT_INDEX)

# ===========================
# REGEXES
# ===========================

# Lines that start with: "1.", "2.*", "3. (a)", "10.", etc.
exercise_line_rx = re.compile(r'^\s*\d+\.\s*')

# Page header: "CHAPTER X - EXERCISES"
exercise_page_header_rx = re.compile(
    r'chapter\s*\d+\s*[^a-zA-Z0-9]{0,4}\s*exerc[a-z]{2,6}',
    re.IGNORECASE
)


# Chapter number inside page
chapter_heading_rx = re.compile(r'\bChapter\s+(\d+)\b', re.IGNORECASE)


# ===========================
# OCR & PAGE HELPERS
# ===========================

def render_page_to_image(page, dpi=DPI):
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def image_to_cv(img_pil):
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def text_boxes_from_ocr(img_pil):
    data = pytesseract.image_to_data(img_pil, lang=OCR_LANG,
                                     output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['text'])
    for i in range(n):
        txt = data['text'][i].strip()
        if not txt:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        boxes.append({
            "text": txt,
            "bbox": (x, y, x+w, y+h)
        })
    return boxes

def merge_boxes_into_lines(boxes, y_tol=12):
    lines = []
    boxes_sorted = sorted(boxes, key=lambda b: b["bbox"][1])

    for b in boxes_sorted:
        x1, y1, x2, y2 = b["bbox"]
        placed = False

        for line in lines:
            ly1, ly2 = line["y1"], line["y2"]
            if abs(y1 - ly1) < y_tol or (y1 <= ly2 + y_tol and y2 >= ly1 - y_tol):
                line["words"].append(b)
                line["y1"] = min(line["y1"], y1)
                line["y2"] = max(line["y2"], y2)
                line["x1"] = min(line["x1"], x1)
                line["x2"] = max(line["x2"], x2)
                placed = True
                break

        if not placed:
            lines.append({
                "words": [b],
                "y1": y1,
                "y2": y2,
                "x1": x1,
                "x2": x2
            })

    out = []
    for line in lines:
        words = sorted(line["words"], key=lambda w: w["bbox"][0])
        text = " ".join(w["text"] for w in words)
        out.append({
            "text": text,
            "bbox": (line["x1"], line["y1"], line["x2"], line["y2"])
        })

    return out


# ===========================
# BLOCK DETECTION
# ===========================

def find_exercise_blocks(lines):
    blocks = []
    i = 0
    while i < len(lines):
        ln = lines[i]["text"].strip()

        if exercise_line_rx.match(ln):
            start = i
            i += 1

            while i < len(lines):
                nxt = lines[i]["text"].strip()

                if exercise_line_rx.match(nxt):
                    break
                if nxt == "":
                    break
                if nxt.lower().startswith("chapter") or nxt.lower().startswith("section"):
                    break

                i += 1

            end = i - 1
            blocks.append((start, end))
        else:
            i += 1

    return blocks


# ===========================
# BOUNDING BOX HELPERS
# ===========================

def bbox_union(bboxes, pad=10, w_limit=None, h_limit=None):
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = x2 + pad
    y2 = y2 + pad

    if w_limit is not None:
        x2 = min(x2, w_limit)
    if h_limit is not None:
        y2 = min(y2, h_limit)

    return (x1, y1, x2, y2)

def smart_bbox_expand(bx, page_img, extra_down=350, extra_right=150, margin=25):
    """
    Expand a text-based bounding box to include diagrams, graphics, tables, etc.

    bx = (x1, y1, x2, y2)
    page_img = PIL image of the whole page
    """

    W, H = page_img.size
    x1, y1, x2, y2 = bx

    # Expand downward (graphics normally sit under text)
    y2 = min(H, y2 + extra_down)

    # Expand rightwards (graphics often align right of text)
    x2 = min(W, x2 + extra_right)

    # Expand padding all around
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W, x2 + margin)
    y2 = min(H, y2 + margin)

    return (x1, y1, x2, y2)



# ===========================
# MAIN EXTRACTION FUNCTION
# ===========================

def extract():
    doc = fitz.open(PDF_PATH)
    index = []

    with Progress(
        TextColumn("[bold blue]{task.description} "),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TimeRemainingColumn(),
    ) as progress:

        page_task = progress.add_task("Pages", total=len(doc))
        total_ex = 0

        for page_number in range(len(doc)):
            page = doc[page_number]

            # -----------------------------------
            # 1. Detect exercise page header
            # -----------------------------------
            # -------------------------------
            # HEADER DETECTION
            # -------------------------------

            # Try reading header from text layer
            header_clip = fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.20)
            header_text = page.get_text("text", clip=header_clip)

            # OCR fallback if no text layer
            if not header_text.strip():
                small_img = render_page_to_image(page)
                top_crop = small_img.crop((0, 0, small_img.width, int(small_img.height * 0.20)))
                header_text = pytesseract.image_to_string(top_crop, lang="eng")

            print("HEADER:", repr(header_text))

            if not exercise_page_header_rx.search(header_text):
                progress.update(page_task, advance=1)
                continue


            # -----------------------------------
            # 2. Render & OCR (only exercise pages)
            # -----------------------------------
            pil = render_page_to_image(page)
            cvimg = image_to_cv(pil)
            h, w = cvimg.shape[:2]

            ocr_task = progress.add_task("  OCR", total=1)
            boxes = text_boxes_from_ocr(pil)
            progress.update(ocr_task, advance=1)
            progress.remove_task(ocr_task)

            if not boxes:
                progress.update(page_task, advance=1)
                continue

            lines = merge_boxes_into_lines(boxes)

            # detect chapter
            chap = None
            m = chapter_heading_rx.search(header_text)
            if m:
                try:
                    chap = int(m.group(1))
                except:
                    pass

            # -----------------------------------
            # 3. Find all exercises on page
            # -----------------------------------
            blocks = find_exercise_blocks(lines)

            ex_task = progress.add_task("  Exercises", total=len(blocks))

            for bidx, (s, e) in enumerate(blocks):
                progress.update(ex_task, advance=1)

                block_bboxes = [lines[i]["bbox"] for i in range(s, e+1)]
                # 1. text-only bounding box
                bx = bbox_union(block_bboxes, pad=12, w_limit=w, h_limit=h)

                # 2. expand to include graphics / figures
                bx = smart_bbox_expand(bx, pil, extra_down=450, extra_right=250, margin=35)

                # 3. crop final region
                crop = pil.crop(bx)


                # Extract exercise number
                first = lines[s]["text"].strip()
                mnum = re.match(r'^(\d+)\.', first)
                ex_num = mnum.group(1) if mnum else f"{page_number}_{s}"

                filename = f"ch{chap if chap else 'x'}_ex{ex_num}_p{page_number+1}_{bidx}.png"
                filepath = os.path.join(OUT_IMAGES, filename)

                crop.save(filepath)

                index.append({
                    "chapter": chap,
                    "exercise": ex_num,
                    "page": page_number+1,
                    "image": f"data/images/{filename}",
                    "bbox": bx
                })

                total_ex += 1

            progress.remove_task(ex_task)
            progress.update(page_task, advance=1)

        # -----------------------------------
        # SAVE INDEX
        # -----------------------------------
        with open(OUT_INDEX, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

        print(f"\nDone: {total_ex} exercises extracted.")
        print(f"Index written to {OUT_INDEX}")


# ===========================
# MAIN ENTRY
# ===========================

if __name__ == "__main__":
    extract()
