from RNN import Encoder, Decoder
import torch
from utils import *
from createChatData import *

device = "cuda" if torch.cuda.is_available() else "cpu"

input_lang, output_lang, pairs = prepareData('data/data.txt')

encoder = Encoder(input_size=input_lang.n_words)
decoder = Decoder(input_size=output_lang.n_words)

encoder.load_state_dict(torch.load("./models/encoder_state_dict.pt"))
decoder.load_state_dict(torch.load("./models/decoder_state_dict.pt"))
encoder.eval()
decoder.eval()


def predict(input_sentence):
    # obtenemos el último estado oculto del encoder
    hidden = encoder(input_sentence.unsqueeze(0))
    # calculamos las salidas del decoder de manera recurrente
    decoder_input = torch.tensor(
        [[output_lang.word2index['SOS']]], device=device)
    # iteramos hasta que el decoder nos de el token <eos>
    outputs = []
    while True:
        output, hidden = decoder(decoder_input, hidden)
        decoder_input = torch.argmax(output, axis=1).view(1, 1)
        outputs.append(decoder_input.cpu().item())
        if decoder_input.item() == output_lang.word2index['EOS']:
            break
    return output_lang.sentenceFromIndex(outputs)


def talk(sentence):
    sn = input_lang.indexesFromSentence(sentence)
    sn = torch.Tensor(sn)
    sn = sn.type(torch.long)
    predicted = predict(sn)
    return predicted


#translate("my name is luis")

def conversation():
    mensaje = input("TU: ")
    mensaje = normalizeString(mensaje)

    respuestaArray = talk(mensaje)
    respuesta = "" 
    for res in respuestaArray:
        if res == "EOS":
            break

        respuesta += (" " + res)
    print(f"Ana-BOT: {respuesta}")
    anaRespuesta = input("Real: ")
    if anaRespuesta != "":
        chatting(mensaje, anaRespuesta)

while True:
    conversation()