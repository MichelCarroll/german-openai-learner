from pydantic import BaseSettings

class AppConfig(BaseSettings):
    openai_key: str
    frequency_list_path: str = 'data/frequency_list.txt'
    vocabulary_size: int = 20
    num_paragraphs: int = 2
