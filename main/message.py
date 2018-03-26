import platform
import os

def say(text):
    print(text)
    if platform.system() != 'Windows':
        os.system('say "' + text + '"')
