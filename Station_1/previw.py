import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from PIL import Image, ImageTk

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
TRASH_NAME = "_aussortiert"


class BildSichter:
    def __init__(self, root, folder):
        self.root = root
        self.folder = folder
        self.trash = folder / TRASH_NAME

        # Alle Bilder suchen (auch in Unterordnern), Papierkorb-Ordner auslassen
        self.files = sorted(
            p for p in folder.rglob("*")
            if p.suffix.lower() in IMG_EXT and TRASH_NAME not in p.parts
        )
        self.index = 0
        self.history = []          # für "rückgängig": (Originalpfad, Papierkorbpfad, Position)
        self.photo = None          # tkinter braucht eine Referenz, sonst verschwindet das Bild
        self._resize_job = None

        # --- Fenster aufbauen ---
        root.title(f"Bild-Sichter – {folder}")
        root.geometry("1000x800")
        root.configure(bg="#222222")

        self.status = tk.Label(root, anchor="w", bg="#111111", fg="#dddddd",
                               font=("Consolas", 11), padx=8, pady=4)
        self.status.pack(side="bottom", fill="x")

        self.help = tk.Label(root, anchor="w", bg="#111111", fg="#888888", font=("Consolas", 10), padx=8,
                             text="→/Leertaste: weiter   ←: zurück   Entf/X: aussortieren   Z: rückgängig   Esc: beenden")
        self.help.pack(side="bottom", fill="x")

        self.view = tk.Label(root, bg="#222222", fg="#ff8888", font=("Arial", 14))
        self.view.pack(fill="both", expand=True)

        # --- Tastenkürzel ---
        for key in ("<Right>", "<space>", "d", "D"):
            root.bind(key, self.next_image)
        for key in ("<Left>", "a", "A"):
            root.bind(key, self.prev_image)
        for key in ("<Delete>", "<BackSpace>", "x", "X"):
            root.bind(key, self.delete_image)
        for key in ("z", "Z"):
            root.bind(key, self.undo)
        for key in ("<Escape>", "q", "Q"):
            root.bind(key, lambda e: root.destroy())

        root.bind("<Configure>", self.on_resize)   # Bild neu skalieren, wenn das Fenster größer/kleiner wird

        root.update_idletasks()
        self.show()

    # ---------------------------------------------------------------
    # Anzeige
    # ---------------------------------------------------------------
    def show(self):
        if not self.files:
            self.view.config(image="", text="Keine Bilder (mehr) in diesem Ordner.")
            self.status.config(text=f"Aussortiert: {len(self.history)}")
            return

        self.index = max(0, min(self.index, len(self.files) - 1))
        path = self.files[self.index]

        try:
            # "with" schließt die Datei sofort wieder – wichtig unter Windows,
            # sonst kann sie danach nicht verschoben werden
            with Image.open(path) as img:
                img.load()
                orig_w, orig_h = img.size
                img = img.convert("RGB")
        except Exception as err:
            self.photo = None
            self.view.config(image="", text=f"Bild kann nicht geöffnet werden (evtl. defekt):\n{err}")
            self.update_status(path, "defekt")
            return

        # an Fenstergröße anpassen (nur verkleinern, kleine Bilder bleiben klein)
        max_w = max(self.view.winfo_width(), 100)
        max_h = max(self.view.winfo_height(), 100)
        img.thumbnail((max_w, max_h))

        self.photo = ImageTk.PhotoImage(img)
        self.view.config(image=self.photo, text="")
        self.update_status(path, f"{orig_w}×{orig_h} px")

    def update_status(self, path, info):
        rel = path.relative_to(self.folder)
        self.status.config(
            text=f"{self.index + 1}/{len(self.files)}   Klasse: {path.parent.name}   "
                 f"{info}   {rel}   |   aussortiert: {len(self.history)}"
        )

    def on_resize(self, event):
        if event.widget is not self.root:
            return
        # kurz warten, damit beim Ziehen am Fensterrand nicht ständig neu gezeichnet wird
        if self._resize_job:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(150, self.show)

    # ---------------------------------------------------------------
    # Navigation
    # ---------------------------------------------------------------
    def next_image(self, event=None):
        if self.index < len(self.files) - 1:
            self.index += 1
            self.show()
        else:
            self.root.bell()   # Ende erreicht

    def prev_image(self, event=None):
        if self.index > 0:
            self.index -= 1
            self.show()
        else:
            self.root.bell()   # Anfang erreicht

    # ---------------------------------------------------------------
    # Aussortieren und Rückgängig
    # ---------------------------------------------------------------
    def delete_image(self, event=None):
        if not self.files:
            return
        path = self.files[self.index]
        target = self.trash / path.relative_to(self.folder)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():   # gleicher Name schon im Papierkorb
            target = target.with_name(f"{target.stem}_{len(self.history)}{target.suffix}")

        try:
            shutil.move(str(path), str(target))
        except OSError as err:
            self.root.bell()
            self.status.config(text=f"Konnte nicht verschieben: {err}")
            return

        self.files.pop(self.index)
        self.history.append((path, target, self.index))
        self.show()   # gleicher Index = automatisch das nächste Bild

    def undo(self, event=None):
        if not self.history:
            self.root.bell()
            return
        original, target, position = self.history.pop()
        original.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(target), str(original))
        self.files.insert(position, original)
        self.index = position
        self.show()


def main():
    root = tk.Tk()

    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        root.withdraw()
        chosen = filedialog.askdirectory(title="Bilderordner auswählen")
        if not chosen:
            return
        folder = Path(chosen)
        root.deiconify()

    if not folder.is_dir():
        print(f"Ordner nicht gefunden: {folder}")
        return

    BildSichter(root, folder.resolve())
    root.mainloop()


if __name__ == "__main__":
    main()