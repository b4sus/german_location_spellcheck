#german location spellcheck
Highly experimental project where I am trying to use some machine learning techniques to have a spellcheck (or autocorrect)
for limited set of words - german locations (cities/towns - Städte/Gemeinden). The idea is that when I misspell the location
I will get list of locations which are 'close' to what I've typed.

First issue was how to vectorize the locations - I used CountVectorizer with char analyzer from sklearn
(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). It first creates
the corpus - set of all characters in all locations (the whole alphabet) and then for
each location it generates vector with count of each item from the corpus. For example 'berlin' will end up as a vector of length
(every location will be a vector of same length (number of characters)) 26 with all 0s except of six 1s for each character 'b',
'e', 'r', 'l', 'i', 'n'.
Those are 1grams - corpus is made of single characters. I also tried 2grams, where each 2 characters are counted - for 'berlin' that
would be 'be', 'er', 'rl', 'li', 'in' - so 1s for these and many 0s - as the corpus is set of all possible character pairs in
all locations (518).

First I tried to see what are the closest (vectorized) locations to my sample misspelled location 
#TODOs
* write this readme
* summary table