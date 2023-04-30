

import os
from bs4 import BeautifulSoup
from pydantic import BaseModel
import requests

URL="https://en.wiktionary.org/wiki/User:Matthias_Buchmeier/German_frequency_list-1-5000"

class FrequencyListEntry(BaseModel):
    word: str
    frequency: int

def retrieve_frequency_list() -> list[FrequencyListEntry]:
    content = requests.get(URL).text
    soup = BeautifulSoup(content, "html.parser")
    space_delimited_list = soup.select('#mw-content-text > div.mw-parser-output > p')[0].text
    word_entry_list = space_delimited_list.strip().split('\n')
    raw_frequency_list = [word_entry.split(' ', 1) for word_entry in word_entry_list]
    return [FrequencyListEntry(word=word, frequency=int(frequency)) for frequency, word in raw_frequency_list]

def save_frequency_list(frequency_list: list[FrequencyListEntry], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as file:
        for entry in frequency_list:
            file.write(f'{entry.frequency} {entry.word}\n')

def open_local_frequency_list(path: str) -> list[FrequencyListEntry]:
    with open(path, 'r') as file:
        return [FrequencyListEntry(frequency=int(line.split(' ')[0]), word=line.split(' ')[1].strip()) for line in file.readlines()]