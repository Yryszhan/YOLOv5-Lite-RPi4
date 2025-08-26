import pyttsx3

engine = pyttsx3.init()
engine.setProperty('voice', 'russian')  # Альтернатива: engine.setProperty('voice', 'ru')

engine.say("Привет! Впереди машина.")
engine.runAndWait()
