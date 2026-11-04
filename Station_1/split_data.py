import random
import shutil
from pathlib import Path

source = Path("data/sharks")
endpath = Path("data/sharks_split")

TRAIN = 0.70
VAL = 0.15

random.seed(42)


def main():
    if endpath.exists():
        shutil.rmtree(endpath)

    klassen_ordner = sorted(
        ordner for ordner in source.iterdir()
        if ordner.is_dir() and not ordner.name.startswith("_")
    )

    for ordner in klassen_ordner:
        bilder = sorted(ordner.glob("*"))
        random.shuffle(bilder)

        anzahl = len(bilder)
        ende_train = int(anzahl * TRAIN)
        ende_val = ende_train + int(anzahl * VAL)

        aufteilung = {
            "train": bilder[:ende_train],
            "val": bilder[ende_train:ende_val],
            "test": bilder[ende_val:],
        }

        for teil, dateien in aufteilung.items():
            ziel_ordner = endpath / teil / ordner.name
            ziel_ordner.mkdir(parents=True, exist_ok=True)
            for datei in dateien:
                shutil.copy(datei, ziel_ordner / datei.name)

        print(f"{ordner.name:20s} train={len(aufteilung['train']):4d}  "
              f"val={len(aufteilung['val']):4d}  test={len(aufteilung['test']):4d}")


if __name__ == "__main__":
    main()
