def addingLine(text, filename = "data"):
    text = text + " \n"
    try:
        file = open(f"data/{filename}.txt", "r")
    except:
        file = open(f"data/{filename}.txt", "w")
        file.close()
        file = open(f"data/{filename}.txt", "r")
    lines = file.readlines()
    file.close()
    
    file = open(f"data/{filename}.txt", "w")
    lines.append(text)
    file.writelines(lines)
    file.close()

def clean_sentence(text):
    text = text.replace("can't", "can not").replace("won't", "will not").replace("n't", " not")
    text = text.replace("'m", " am").replace("'re", " are").replace("'s", "  is").replace("'d", " would")
    text = text.replace("'ll", " will").replace("'t", " not").replace("'ve", "  have")
    return text

def chatting(msg1, msg2, filename = "data"):
    msg1 = clean_sentence(msg1)
    msg2 = clean_sentence(msg2)

    text = msg1 + "\t" + msg2
    addingLine(text, filename)