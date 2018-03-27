import platform
import os

def say(text):
    print(text)
    if platform.system() == 'Darwin':
        os.system('say "' + text + '"')
