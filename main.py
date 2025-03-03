import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify GROQ API key is loaded
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please check your .env file.")

# Create necessary directories - THIS MUST BE FIRST
os.makedirs("data/tmp", exist_ok=True)
os.makedirs("data/audio", exist_ok=True)
os.makedirs("data/html", exist_ok=True)
os.makedirs("data/database", exist_ok=True)  # for SQLite database

# Set DATA_DIR environment variable
os.environ["DATA_DIR"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# After setting DATA_DIR
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY not found in environment variables. Please check your .env file.")
os.environ["COHERE_API_KEY"] = COHERE_API_KEY

# Now we can do the rest of the imports
import sys
import asyncio
from modules.vocalize.async_edgetts import textToSpeechBytes, AudioPlayer

IMPORTS_SOUND = r"data\audio\imports.wav"
AFTER_IMPORTS_SOUND = r"data\audio\after_imports.wav"

audioPlayer = AudioPlayer()
audioPlayer.setVolume(0.2)
audioPlayer.play(open(IMPORTS_SOUND, 'rb').read())

import subprocess
from modules.sqlqueue import SqlQueue

SR = "async_js_sr"
SR_DATABASE = rf"data\tmp\{SR}.queue.db"
SR_PROCESS = subprocess.Popen([f"{sys.executable}", f"modules/vocalize/{SR}.py"])
SR_QUEUE = SqlQueue(SR_DATABASE)

import time
from util.jarvis.main import process, JFunction, codeBrew, historyDb
from modules.prompt.base import Role
from typing import Optional

audioPlayer.play(open(AFTER_IMPORTS_SOUND, 'rb').read())

audioPlayer.setVolume(1)

def textToAudioPrint(*args, **kwargs):
    string = " ".join([str(x) for x in args])
    
    print(f"{string = }")
    
    historyDb.addMessage(
        role = Role.assistant.value,
        content = string
    )

    print(f"{string = }")
    audioPlayer.play(
        asyncio.run(
            textToSpeechBytes(string)
        ),
    )
    while audioPlayer.isPlaying() and kwargs.get('block', False):
        time.sleep(0.1)

async def asyncTextToAudioPrint(*args, **kwargs):
    string = " ".join([str(x) for x in args])

    print(f"{string = }")
    
    historyDb.addMessage(
        role = Role.assistant.value,
        content = string
    )

    print(f"{string = }")
    audioPlayer.play(await textToSpeechBytes(string))
    
    while audioPlayer.isPlaying() and kwargs.get('block', False):
        time.sleep(0.1)

def vocalizeInput(prompt: Optional[str] = "", ignoreThreshold: float = 0.3) -> str:
    if prompt:
        textToAudioPrint(prompt, block=True)
    SR_QUEUE.clear()
    
    # ignoreThreshold is the time to wait for the user to speak
    SR_QUEUE.get(timeout=ignoreThreshold)
    
    recordedAudio: str = SR_QUEUE.get()
    historyDb.addMessage(
        role = Role.user.value,
        content = recordedAudio
    )
    return recordedAudio

# Override print function
codeBrew.print = textToAudioPrint
codeBrew.input = vocalizeInput

async def jFunctionEval(jFunctions: list[JFunction]) -> list[str | None]:
    taskList = []
    for jFunction in jFunctions:
        taskList.append(asyncio.to_thread(jFunction.function))
    return await asyncio.gather(*taskList)

async def main():
    audioPlayer.play(await textToSpeechBytes("Jarvis Online!, Hello. How can I help you?"))
    while True:
        query = vocalizeInput()
        result = await process(query)
        dmm = result['dmm']
        edmm = result['edmm']
        ndmm = result['ndmm']
        timeTaken = result['timeTaken']
        
        print(f"""{dmm = }\n{edmm = }\n{ndmm = }\n{timeTaken = }""")

        await asyncio.gather(
            jFunctionEval(ndmm),
            jFunctionEval(edmm)
        )

        dmmResult = await jFunctionEval(dmm)
        # Clean up the results to handle multiline strings
        strOnlyDmmResult = [
            str(x).replace('\n', ' ').strip() 
            for x in dmmResult 
            if isinstance(x, str)
        ]
        if len(strOnlyDmmResult) == 0:
            continue

        while audioPlayer.isPlaying(): # wait for audio to finish playing by CodeBrew
            await asyncio.sleep(0.1)
        await asyncTextToAudioPrint(*strOnlyDmmResult, block=True)

def get_desktop_path():
    base = os.path.join(os.environ['USERPROFILE'])
    standard_desktop = os.path.join(base, 'Desktop')
    onedrive_desktop = os.path.join(base, 'OneDrive', 'Desktop')
    
    if os.path.exists(onedrive_desktop):
        return onedrive_desktop
    elif os.path.exists(standard_desktop):
        return standard_desktop
    else:
        raise FileNotFoundError("Could not find Desktop folder")

# Get the correct path to the desktop
DESKTOP_PATH = get_desktop_path()  # Use the get_desktop_path() function instead of hardcoding
os.environ["DESKTOP_PATH"] = DESKTOP_PATH

# Check if the desktop path exists
if not os.path.exists(DESKTOP_PATH):
    raise FileNotFoundError(f"Desktop path not found: {DESKTOP_PATH}")

# Initialize a counter for shortcuts
shortcut_count = 0

if __name__ == "__main__":
    asyncio.run(main())


