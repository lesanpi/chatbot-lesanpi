

def addingLine(text):
    text = text + " \n"
    file = open("data/data.txt", "r")
    lines = file.readlines()
    file.close()
    
    file = open("data/data.txt", "w")
    lines.append(text)
    file.writelines(lines)
    file.close()

def chatting(msg1, msg2):
    text = msg1 + "\t" + msg2
    addingLine(text)

name_pj1 = input("Ingresa nombre de la primera persona en hablar: ")
name_pj2 = input("Ingresa nombre de la segunda persona en hablar: ")
print()
while True:
    msg1 = input(name_pj1.upper() + ": ")
    msg2 = input(name_pj2.upper() + ": ")

    chatting(msg1, msg2)