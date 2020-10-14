from createChatData import *

name_pj1 = input("Ingresa nombre de la primera persona en hablar: ")
name_pj2 = input("Ingresa nombre de la segunda persona en hablar: ")
print()
while True:
    msg1 = input(name_pj1.upper() + ": ")
    if msg1 == "exit()":
        break
    msg2 = input(name_pj2.upper() + ": ")
    if msg2 == "exit()":
        break
    chatting(msg1, msg2)