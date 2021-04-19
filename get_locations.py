import re

import requests
from bs4 import BeautifulSoup


def load_and_preprocess_locations():
    locations = []
    with open("data/locations", encoding="utf-8") as locations_file:
        for loc in locations_file:
            locations.append(preprocess(loc))
    return locations


def preprocess(word):
    word = word.strip().lower()
    word = re.sub("\s", "", word)
    word = re.sub("\"", "", word)
    word = re.sub("\)", "", word)
    word = re.sub("\(", "", word)
    word = re.sub("\.", "", word)
    word = re.sub(",", "", word)
    word = re.sub("-", "", word)
    word = re.sub("/", "", word)
    word = re.sub("ä", "ae", word)
    word = re.sub("ö", "oe", word)
    word = re.sub("ü", "ue", word)
    word = re.sub("ß", "ss", word)
    return word


def fetch_all_locations():
    letters = (chr(code) for code in range(65, 91))
    locations = []
    for letter in letters:
        locations.extend(fetch_locations_for_letter(letter))
    return locations


def fetch_locations_for_letter(letter):
    print(f"getting locations for letter {letter}")
    soup = BeautifulSoup(requests.get("http://www.deutsche-staedte.de/staedte.php?city=" + letter).content)
    ul = soup.find("ul", style="list-style-type: square; list-style-position: outside; padding-left: 25px;")
    return (location_link.text.strip() for location_link in ul.find_all("a") if location_link.text.strip() != "")


if __name__ == "__main__":
    locations = fetch_all_locations()
    with open("data/locations", mode="w", encoding="UTF-8") as locations_file:
        for loc in locations:
            if loc.strip() != "":
                locations_file.write(f"{loc}\n")
