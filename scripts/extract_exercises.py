#!/usr/bin/env python3
import os
import json
import pytesseract
import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import ImageGrab, Image, ImageTk
import subprocess
from PIL import Image
import io

# ===========================
# CONFIGURATION
# ===========================

OCR_LANG = "eng"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

OUT_IMAGES = os.path.join(ROOT, "site", "data", "images")
OUT_INDEX = os.path.join(ROOT, "site", "data", "index.json")

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(os.path.dirname(OUT_INDEX), exist_ok=True)

CHAPTER_EXERCISE_COUNTS = {
    1: 21,
    2: 18,
    3: 18,
    4: 14,
    5: 27,
    6: 24,
    7: 27,
    8: 22,
    9: 30,
    10: 24,
    11: 14,
    12: 17,
    13: 19,
    14: 17,
    15: 10,
    16: 6,
    17: 9,
    18: 6,
    19: 13,
    20: 9,
}

# Load index
if os.path.exists(OUT_INDEX):
    with open(OUT_INDEX, "r", encoding="utf-8") as f:
        index = json.load(f)
else:
    index = []


def save_index():
    with open(OUT_INDEX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def do_ocr(img):
    return pytesseract.image_to_string(img, lang=OCR_LANG)


def save_exercise_image(chapter, exercise, pil_img):
    filename = f"ch{chapter}_ex{exercise}.png"
    filepath = os.path.join(OUT_IMAGES, filename)
    pil_img.save(filepath)
    return filename


def get_next_exercise(chapter):
    used = [int(item["exercise"]) for item in index if item["chapter"] == chapter]
    if not used:
        return 1
    return max(used) + 1


def next_slot():
    """Return (chapter, exercise) of next available slot."""
    for chap, max_ex in CHAPTER_EXERCISE_COUNTS.items():
        nxt = get_next_exercise(chap)
        if nxt <= max_ex:
            return chap, nxt
    return None, None


def grab_clipboard_image():
    """
    Linux version: uses xclip to pull image/png from clipboard.
    Returns PIL image or None.
    """
    try:
        p = subprocess.Popen(
            ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        data, _ = p.communicate()
        if not data:
            return None
        return Image.open(io.BytesIO(data))
    except Exception:
        return None


class CollectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Exercise Collector")

        # Labels
        tk.Label(self.root, text="Chapter:").grid(row=0, column=0, sticky="e")
        tk.Label(self.root, text="Exercise:").grid(row=1, column=0, sticky="e")

        # Chapter/exercise fields (auto-updated)
        self.chapter_entry = tk.Entry(self.root, width=6)
        self.exercise_entry = tk.Entry(self.root, width=6)
        self.chapter_entry.grid(row=0, column=1, padx=10, pady=5)
        self.exercise_entry.grid(row=1, column=1, padx=10, pady=5)

        # When chapter changes → update exercise only
        self.chapter_entry.bind("<KeyRelease>", self.update_exercise_only)

        # Paste button
        self.paste_button = tk.Button(
            self.root, text="Paste Screenshot (Ctrl+V)", command=self.handle_paste
        )
        self.paste_button.grid(row=2, column=0, columnspan=2, pady=10)

        # Preview
        self.preview_label = tk.Label(self.root)
        self.preview_label.grid(row=3, column=0, columnspan=2)

        # Bind Ctrl+V
        self.root.bind("<Control-v>", lambda e: self.handle_paste())

        # Initialize fields with next available
        self.set_next_available()

        self.root.mainloop()

    # ------------------------------------------

    def set_next_available(self):
        chap, ex = next_slot()
        self.chapter_entry.delete(0, tk.END)
        self.exercise_entry.delete(0, tk.END)

        if chap is None:
            self.chapter_entry.insert(0, "DONE")
            self.exercise_entry.insert(0, "DONE")
        else:
            self.chapter_entry.insert(0, str(chap))
            self.exercise_entry.insert(0, str(ex))

    # ------------------------------------------

    def update_exercise_only(self, event=None):
        """Update exercise when chapter changes, but do not modify chapter."""
        text = self.chapter_entry.get().strip()
        if not text.isdigit():
            return
        chap = int(text)
        if chap not in CHAPTER_EXERCISE_COUNTS:
            return

        ex = get_next_exercise(chap)
        if ex > CHAPTER_EXERCISE_COUNTS[chap]:
            self.exercise_entry.delete(0, tk.END)
            self.exercise_entry.insert(0, "FULL")
        else:
            self.exercise_entry.delete(0, tk.END)
            self.exercise_entry.insert(0, str(ex))

    # ------------------------------------------

    def handle_paste(self):
        try:
            img = grab_clipboard_image()

        except Exception:
            img = None

        if img is None:
            messagebox.showerror("Error", "Clipboard does not contain an image.")
            return

        chapter_str = self.chapter_entry.get()
        ex_str = self.exercise_entry.get()

        if not chapter_str.isdigit() or not ex_str.isdigit():
            messagebox.showerror("Error", "No available slot.")
            return

        chapter = int(chapter_str)
        exercise = int(ex_str)

        # Show preview
        prev = img.copy()
        prev.thumbnail((300, 300))
        self.tk_preview = ImageTk.PhotoImage(prev)
        self.preview_label.config(image=self.tk_preview)

        # OCR
        ocr = do_ocr(img)

        # Build correct filename
        # Example: ch2_ex8_p58_0.png
        file_id = 0
        filename = f"ch{chapter}_ex{exercise}_{file_id}.png"
        filepath = os.path.join(OUT_IMAGES, filename)

        # Avoid accidental overwrite: increment _0, _1, _2...
        while os.path.exists(filepath):
            file_id += 1
            filename = f"ch{chapter}_ex{exercise}_{file_id}.png"
            filepath = os.path.join(OUT_IMAGES, filename)

        # Save image
        img.save(filepath)

        # Append to index
        index.append(
            {
                "chapter": chapter,
                "exercise": str(exercise),
                "ocr": ocr,
                "image": f"data/images/{filename}",
            }
        )

        save_index()

        messagebox.showinfo("Saved", f"Saved: {filename}")

        # Move to next available slot
        self.set_next_available()


if __name__ == "__main__":
    CollectorGUI()
