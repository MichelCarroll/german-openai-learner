
from dotenv import load_dotenv
import openai

from config import AppConfig
from story_generator import generate_exercise


if __name__ == '__main__':
    load_dotenv()
    config = AppConfig()
    openai.api_key = config.openai_key 
    exercise = generate_exercise(index_from=100, vocabulary_size=config.vocabulary_size)
    print(exercise)

