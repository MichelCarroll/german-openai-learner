import click
from enum import Enum
from dotenv import load_dotenv
from pydantic import BaseSettings, BaseModel
import openai
from spacy_download import load_spacy
from helpers.spacy import PartOfSpeech

from frequency_list import save_frequency_list, open_local_frequency_list, retrieve_frequency_list, FrequencyListEntry


class AppConfig(BaseSettings):
    openai_key: str
    frequency_list_path: str = 'data/frequency_list.txt'
    vocabulary_size: int = 20
    num_paragraphs: int = 2

@click.command()
@click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(count, name):
    """Simple program that greets NAME for a total of COUNT times."""
    for x in range(count):
        click.echo(f"Hello {name}!")

def load_frequency_list() -> list[FrequencyListEntry]:
    try:
        frequency_list = open_local_frequency_list(path=AppConfig().frequency_list_path)
    except FileNotFoundError:
        frequency_list = retrieve_frequency_list()
        save_frequency_list(frequency_list=frequency_list, path= AppConfig().frequency_list_path)
    return frequency_list

def view_parts_of_speech():
    frequency_list = load_frequency_list()
    
    nlp = load_spacy("de_core_news_sm")  

    doc = nlp.pipe("Ich")
    
    for entry in frequency_list[0:100]:
        doc = nlp(entry.word)
        pos = PartOfSpeech(doc[0].pos_)
        print(entry.word, pos.value)

class OpenAIChatCompletionModel(Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"

class Exercise(BaseModel):
    german_text: str 
    english_text: str
    word_list: list[tuple[str, str]]


def generate_exercise(index_from: int, vocabulary_size: int) -> Exercise:
    frequency_list = load_frequency_list()
    words_to_include = [entry.word for entry in frequency_list[index_from:(index_from+vocabulary_size)]]

    translation_prompt = f"""
        {', '.join(words_to_include)}
        ----------------------------------------
        Translate the above words into English, separating each word with a comma:
        """

    response = openai.ChatCompletion.create(
        model=OpenAIChatCompletionModel.GPT_3_5_TURBO.value,
        messages=[
            {"role": "user", "content": translation_prompt}
        ],
        max_tokens=1000,
        temperature=0.9,
    )
    translated_words = response['choices'][0]['message']['content']
    translated_word_list = [word.strip('.').strip() for word in translated_words.split(',')]
    
    word_list_translation = list(zip(words_to_include, translated_word_list))

    num_paragraphs = AppConfig().num_paragraphs
    prompt = (
        f"""Write me {num_paragraphs} paragraphs on a simple subject in German."""
        """Use only basic vocabulary and grammar, geared towards a beginner-level German learner."""
        f"""Use the following word list at least once: ({', '.join(words_to_include)})"""
    )
    response = openai.ChatCompletion.create(
        model=OpenAIChatCompletionModel.GPT_3_5_TURBO.value,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.9,
    )

    short_story_text = response['choices'][0]['message']['content']

    
    translation_prompt = f"""
        {short_story_text}
        ----------------------------------------
        Translate the above text into English:
        """

    response = openai.ChatCompletion.create(
        model=OpenAIChatCompletionModel.GPT_3_5_TURBO.value,
        messages=[
            {"role": "user", "content": translation_prompt}
        ],
        max_tokens=1000,
        temperature=0.9,
    )
    translated_short_story_text = response['choices'][0]['message']['content']
    exercise = Exercise(german_text=short_story_text, english_text=translated_short_story_text, word_list=word_list_translation)
    return exercise



if __name__ == '__main__':
    load_dotenv()
    config = AppConfig()
    openai.api_key = config.openai_key 
    exercise = generate_exercise(index_from=100, vocabulary_size=config.vocabulary_size)
    print(exercise)

