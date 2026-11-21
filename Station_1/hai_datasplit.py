import matplotlib.pyplot as plt

# Haiarten mit Anzahl Bilder (aus Haiarten_Datasplit.xlsx)
haie = {
    "Weißer Hai (white)": 128,
    "Walhai (whale)": 126,
    "Hammerhai (hammerhead)": 118,
    "Weißspitzenhai (whitetip)": 113,
    "Fuchshai (thresher)": 99,
    "Blauhai (blue)": 94,
    "Tigerhai (tiger)": 91,
    "Bullenhai (bull)": 89,
    "Sandtigerhai (sand tiger)": 86,
    "Makohai (mako)": 83,
    "Schwarzspitzenhai (blacktip)": 77,
    "Zitronenhai (lemon)": 76,
    "Riesenhai (basking)": 69,
    "Ammenhai (nurse)": 62,
}

# Aufteilung: 70 % Training, 15 % Validierung, Test bekommt den Rest
namen, training, validierung, test, summen = [], [], [], [], []
for name, anzahl in haie.items():
    tr = round(anzahl * 0.70)
    va = round(anzahl * 0.15)
    te = anzahl - tr - va
    namen.append(name)
    training.append(tr)
    validierung.append(va)
    test.append(te)
    summen.append(anzahl)
    print(f"{name:30s} Summe {anzahl:4d}  Training {tr:4d}  Validierung {va:3d}  Test {te:3d}")

# Größter Hai oben: nach Summe sortieren
reihenfolge = sorted(range(len(namen)), key=lambda i: summen[i])
namen = [namen[i] for i in reihenfolge]
training = [training[i] for i in reihenfolge]
validierung = [validierung[i] for i in reihenfolge]
test = [test[i] for i in reihenfolge]
summen = [summen[i] for i in reihenfolge]

# Gestapeltes waagerechtes Balkendiagramm
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(6.9, 3.6),layout="constrained")
ax.barh(namen, training, color="tab:blue", label="Training (70 %)")
ax.barh(namen, validierung, left=training, color="tab:orange", label="Validierung (15 %)")
links_test = [t + v for t, v in zip(training, validierung)]
ax.barh(namen, test, left=links_test, color="tab:green", label="Test (15 %)")

# Summe hinter jedem Balken
for i, summe in enumerate(summen):
    ax.text(summe + 1, i, str(summe), va="center")

ax.set_xlabel("Anzahl Bilder")
ax.set_xlim(0, max(summen) * 1.1)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
plt.tight_layout()
plt.savefig("hai_datasplit.png", dpi=150)
plt.show()
