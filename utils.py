import unicodedata
import re
from Voc import Voc

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

def filterPair(p):
    # input sequences need to preserve the last word for EOS_token
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH

def prepareData(file, filters=None, max_length=None, reverse=False):

    pairs = read_file(file, reverse)
    print(f"Tenemos {len(pairs)} pares de frases")

    if filters is not None:
        assert max_length is not None
        pairs = [filterPairs(pair) for pair in pairs]
        print(f"Filtramos a {len(pairs)} pares de frases")

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Voc('luis')
        output_lang = Voc('ana')
    else:
        input_lang = Voc('ana')
        output_lang = Voc('luis')

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])

        # add <eos> token
        pair[0] += " EOS"
        pair[1] += " EOS"

    print("Longitud vocabularios:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)

    return input_lang, output_lang, pairs

