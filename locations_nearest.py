from utils import CharCountVectorizer, suggest_locations

if __name__ == "__main__":
    suggester = suggest_locations(CharCountVectorizer(1))
    next(suggester)
    print(suggester.send("hechngeli"))
    print(suggester.send("belir"))
