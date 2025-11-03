#!/bin/sh
echo "Installation des hooks Git..."
cp hooks/* .git/hooks/
chmod +x .git/hooks/*

# Vérifie que Jupytext est installé
if ! python -c "import jupytext" &> /dev/null; then
    echo "Jupytext n'est pas installé. Installation en cours..."
    pip install jupytext
fi

# Création des fichiers .ipynb manquants
echo "Création des fichiers .ipynb manquants..."
for py_file in *_ntb.py; do
    ipynb_file="${py_file/_ntb.py/.ipynb}"
    if [ ! -f "$ipynb_file" ]; then
        echo "Création de $ipynb_file..."
        python -m jupytext --to notebook "$py_file" --output "$ipynb_file"
    else
        echo "$ipynb_file existe déjà."
    fi
done
echo "Setup terminé ! Les hooks et les .ipynb sont prêts."