import json, re

with open("notebooks/donnees_propres.json", "r") as f:
    donnees_old = json.load(f)

def chunking_avec_overlap(texte, taille=512, overlap=50):
    mots = texte.split()
    chunks = []
    i = 0
    while i < len(mots):
        chunk = mots[i : i + taille]
        chunks.append(' '.join(chunk))
        i += taille - overlap
    return chunks

donnees_new = []
for ex in donnees_old:
    texte_complet = ' '.join(ex['chunks'])
    new_chunks = chunking_avec_overlap(texte_complet)
    donnees_new.append({
        "chunks": new_chunks,
        "resume": ex["resume"],
        "nb_chunks": len(new_chunks)
    })

with open("notebooks/donnees_propres.json", "w") as f:
    json.dump(donnees_new, f)

print("OK donnees_propres.json regenere avec overlap")
print(f"   Avant : {donnees_old[0]['nb_chunks']} chunks")
print(f"   Apres : {donnees_new[0]['nb_chunks']} chunks")
