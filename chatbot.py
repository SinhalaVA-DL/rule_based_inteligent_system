import random
import json
import pickle
import numpy as np
from model.chatbot import predictor
from functions.functions import *


from tensorflow import keras

intents = json.loads(open('data\intents.json', encoding="utf8").read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = keras.models.load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = sentence.split()
    sentence_words = [word.lower() for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


while True:
    inp = input("You: ")
    if inp.lower() == "quit":
        break
    results = predict_class(inp)
    tag = results[0]['intent']
    proberbility = results[0]['probability']
    if float(proberbility) == 1:
        list_of_intents = intents['intents']
        for i in list_of_intents:
            if(i['tag'] == tag):
                if i['has_function']:
                    result = get_func[tag]()
                else:
                    result = random.choice(i['responses'])
                break
        print("Bot: ", result)
    else:
        result = predictor.predict(inp)
        print("Bot: ", result)


