import os, sys
sys.path.append(os.getcwd())
from modules.database.sq_dict import SQLiteDict
from modules.database.text_store import TextStore
from modules.llm.base import ModelType
from modules.prompt.base import Prompt, Role, Text, File, Function

from modules.codebrew import codebrewPrompt, samplePrompt, CodeBrew
from modules.database.chat_history import ChatHistoryDB

from modules.music_player.main import MusicPlayer

# ------------------------------------------------ #
from modules.llm._cohere import Cohere, COMMAND_R_PLUS
from modules.llm._groq import Groq, LLAMA_32_90B_VISION_PREVIEW, LLAMA_32_90B_TEXT_PREVIEW
# ------------------------------------------------ #

# Set default values for environment variables
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.getcwd(), "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Create all necessary directories
os.makedirs(os.path.join(DATA_DIR, "sql"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "personality"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "log"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "music_player", "downloads"), exist_ok=True)

# CHEAT CODE
cheatCode = os.getenv("CHEAT_CODE", None)

# SCREENSHOT
SCREENSHOT = os.getenv("SCREENSHOT", "False").lower() == "true"

# Define the database directory and create it if it doesn't exist
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
os.makedirs(DB_DIR, exist_ok=True)

# Update paths with proper defaults
sqDictPath = os.path.join(DB_DIR, "sq_dict.db")
userNotebookPath = os.path.join(DATA_DIR, "sql", "user_notebook.sql")
profile = os.path.join(DATA_DIR, "personality", "humor_jarvis.json")

sqDict = SQLiteDict(sqDictPath)

userNotebook = TextStore(userNotebookPath)

# Jarvis profile found in data/personality/
profile = rf"{os.getenv('DATA_DIR')}\personality\humor_jarvis.json"

# Jarvis DMM LLMs
dmmLLM = Cohere(
    COMMAND_R_PLUS, 
    maxTokens=4096, 
    logFile=os.path.join(DATA_DIR, "log", "dmm.log"), 
    cheatCode=cheatCode
)
edmmLLM = Groq(LLAMA_32_90B_TEXT_PREVIEW, maxTokens=4096, logFile=os.getenv("DATA_DIR") + r"/log/eddm.log", cheatCode=cheatCode)
ndmmLLM = Groq(LLAMA_32_90B_TEXT_PREVIEW, maxTokens=4096, logFile=os.getenv("DATA_DIR") + r"/log/nddm.log", cheatCode=cheatCode)

# CodeBrew LLM prompt template
codeBrewPromptTemplate = Prompt(
    role=Role.system,
    template=[
        Function(codebrewPrompt),
        Text("\n\nHere are some example interactions:"),
        Function(samplePrompt)
    ],
    separator="\n\n"
)

# Update the codeBrewLLM initialization
codeBrewLLM = Groq(
    LLAMA_32_90B_TEXT_PREVIEW, 
    maxTokens=4096, 
    systemPrompt=codeBrewPromptTemplate(),
    logFile=os.getenv("DATA_DIR") + r"/log/codebrew.log", 
    cheatCode=cheatCode
)

codeBrew = CodeBrew(codeBrewLLM, keepHistory=False, verbose=True)


chatBotLLM = Groq(LLAMA_32_90B_VISION_PREVIEW, maxTokens=4096, logFile=os.getenv("DATA_DIR") + r"/log/chatbot.log", cheatCode=cheatCode)


historyDb = ChatHistoryDB(rf"{os.getenv('DATA_DIR')}\sql\chat_history.sql")

dmmHistoryDb = ChatHistoryDB(rf"{os.getenv('DATA_DIR')}\sql\dmm.sql")
edmmHistoryDb = ChatHistoryDB(rf"{os.getenv('DATA_DIR')}\sql\edmm.sql")
ndmmHistoryDb = ChatHistoryDB(rf"{os.getenv('DATA_DIR')}\sql\ndmm.sql")


musicPlayer = MusicPlayer(
    rf"{os.getenv('DATA_DIR')}\music_player\downloads",
    default_volume=float(os.getenv("DEFAULT_MUSIC_PLAYER_VOLUME", 0.1))
)

musicPlayerDmmLLM = Groq(
    LLAMA_32_90B_TEXT_PREVIEW, 
    maxTokens=4096,
    logFile=os.getenv("DATA_DIR") + r"/log/music_player_dmm.log", 
    cheatCode=cheatCode
)