import unicodedata
import re

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    # Normalizar
    s = unicodeToAscii(s.lower().strip())
    # Le da un espacio a los signos de puntacion, exclamacion
    # "a." -> "a ."
    s = re.sub(r"([.!?])", r" \1", s)
    
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def read_file(file, reverse=False):
    # Abro el archivo, separo por saltos de linea
    lines = open(file, encoding='utf-8').read().strip().split('\n')

    # Por cada linea
    # Separo por las tabulaciones \t, agarro las 2 primeras
    pairs = [[normalizeString(s) for s in l.split('\t')[:2]] for l in lines]

    return pairs


pairs = read_file("data/data.txt")
print(pairs)