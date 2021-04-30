#german location spellcheck
Highly experimental project where I am trying to use some techniques from machine learning to have a spellcheck (or autocorrect)
for limited set of words - german locations (cities/towns - Städte/Gemeinden). The idea is that when I misspell the location
I will get list of locations which are 'close' to what I've typed.

Main issue was how to vectorize the locations - I used CountVectorizer with char analyzer from sklearn
(https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html). It first creates
the corpus - set of all characters in all locations (practically the whole alphabet) and then for
each location it generates vector with count of each item from the corpus. For example 'stuttgart' will end up as a vector of length
(every location will be a vector of same length (number of characters)) 26 like this:

[1 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 1 4 1 0 0 0 0 0]

Those are 1grams (the technique is called n-gram) - corpus is made of single characters. I also tried 2grams, where each 2
characters are counted - for 'stuttgart' that would be 'st', 'tu', 'ut', 'tt', 'tg', 'ga', 'ar', 'rt' - so 1s for these
and many 0s - as the corpus is set of all possible character pairs in all locations (518).

Once I had the vectors, I tried to see what are the nearest (vectorized) locations to my sample misspelled location.
Then I wondered how will it change when I use PCA to lower the dimension of the vector. Last I tried to use self-organizing map (som)
which is also sort of dimensionality reduction to see how the cities will align to each other (see https://github.com/b4sus/self-organizing-map).
Now follows the table with few examples:

|word|suggester|suggestions|
|----------|----------|------------------------------------------------------------------------------------------------------------------------------------------|
|hechngeli|nearest_1g|['gleichen', 'elchingen', 'heuchlingen', 'hechingen', 'hecklingen', 'gevenich', 'gelchsheim', 'heilenbach', 'schliengen', 'schleching']|
|hechngeli|nearest_2g|['echem', 'becheln', 'hechingen', 'hellingen', 'hecklingen', 'schelklingen', 'igel', 'rech', 'bengel', 'ellingen']|
|hechngeli|nearest_12g|['hechingen', 'becheln', 'elchingen', 'hecklingen', 'heuchlingen', 'hellingen', 'hetlingen', 'eichen', 'gleichen', 'schelklingen']|
|hechngeli|pca_1g|['gleichen', 'schlieben', 'elchingen', 'reichling', 'hechingen', 'schierling', 'schliengen', 'laichingen', 'neichen', 'liesenich']|
|hechngeli|pca_2g|['boechingen', 'hechingen', 'rechlin', 'wechingen', 'juelich', 'rickling', 'dieblich', 'heuchlingen', 'reichling', 'plochingen']|
|hechngeli|pca_12g|['hechingen', 'reichling', 'rechlin', 'gleichen', 'elchingen', 'schliengen', 'heuchlingen', 'suenching', 'boechingen', 'rainlech']|
|hechngeli|som_1g|['elchingen', 'flechtingen', 'gleichen', 'golchen', 'loiching', 'olching', 'hechingen', 'spaichingen', 'clingen', 'creglingen']|
|hechngeli|som_2g|['hahnbach', 'hahnenbach', 'schneckenhausen', 'haehnichen', 'neuhausschierschnitz', 'schneizlreuth', 'teuschnitz', 'schneppenbach', 'hellenhahnschellenberg', 'badzwischenahn']|
|belir|nearest_1g|['berlin', 'birgel', 'reil', 'briedel', 'lisberg', 'irrel', 'rabel', 'breit', 'verl', 'belm']|
|belir|nearest_2g|['belm', 'belg', 'elbe', 'beelitz', 'jabel', 'belum', 'elben', 'rabel', 'irrel', 'elz']|
|belir|nearest_12g|['elbe', 'irrel', 'rabel', 'belg', 'birgel', 'belm', 'berlin', 'bermel', 'belau', 'elz']|
|belir|pca_1g|['berlin', 'elbe', 'irrel', 'brilon', 'nebel', 'elben', 'birgel', 'reil', 'biblis', 'beelen']|
|belir|pca_2g|['irrel', 'firrel', 'kiel', 'wirges', 'birgel', 'kist', 'hirten', 'nebel', 'pirna', 'rerik']|
|belir|pca_12g|['irrel', 'firrel', 'birgel', 'kiel', 'nebel', 'belg', 'reinbek', 'reil', 'gielert', 'berlin']|
|belir|som_1g|['beelen', 'elbe', 'elben', 'elleben', 'nebel', 'albessen', 'selb', 'belm', 'bermel', 'brehme']|
|belir|som_2g|['dabel', 'dobel', 'putbus', 'elben', 'selb', 'baabe', 'cottbus', 'elbe', 'oberelbert', 'werbenelbe']|
|stuttgart|nearest_1g|['stuttgart', 'grettstadt', 'trittau', 'rastatt', 'mutterstadt', 'buttstaedt', 'unstruttal', 'otterstadt', 'tarmstedt', 'tastrup']|
|stuttgart|nearest_2g|['stuttgart', 'putgarten', 'gartow', 'stuer', 'marth', 'tutow', 'utarp', 'buttlar', 'stuhr', 'barth']|
|stuttgart|nearest_12g|['stuttgart', 'rastatt', 'putgarten', 'hattert', 'trittau', 'astert', 'wittgert', 'mutterstadt', 'buttstaedt', 'grettstadt']|
|stuttgart|pca_1g|['stuttgart', 'buettstedt', 'buttstaedt', 'grettstadt', 'ettenstatt', 'uttenreuth', 'trittau', 'rastatt', 'unstruttal', 'buttelstedt']|
|stuttgart|pca_2g|['stuttgart', 'putgarten', 'holtgast', 'nuthetal', 'mittenaar', 'etgert', 'glottertal', 'hattert', 'uttenreuth', 'rottaminn']|
|stuttgart|pca_12g|['stuttgart', 'rastatt', 'abstatt', 'putgarten', 'gusterath', 'hattert', 'glottertal', 'nastaetten', 'retterath', 'ottrau']|
|stuttgart|som_1g|['althengstett', 'ettenstatt', 'grettstadt', 'hattstedt', 'hettstadt', 'stuttgart', 'buettstedt', 'buttelstedt', 'buttstaedt', 'parkstetten']|
|stuttgart|som_2g|['stuttgart', 'tutow', 'alttucheband', 'altusried', 'stutensee', 'landstuhl', 'stuhr', 'tutzing', 'stuvenborn', 'tuchenbach']|
|stutgart|nearest_1g|['stuttgart', 'trittau', 'gusterath', 'rastatt', 'tastrup', 'baustert', 'unstruttal', 'gruenstadt', 'garstedt', 'troestau']|
|stutgart|nearest_2g|['stuttgart', 'putgarten', 'gartow', 'barth', 'tutow', 'utarp', 'stuhr', 'marth', 'stuer', 'sagard']|
|stutgart|nearest_12g|['stuttgart', 'putgarten', 'garstedt', 'astert', 'stuer', 'tutow', 'lautert', 'gartow', 'baustert', 'staudt']|
|stutgart|pca_1g|['stuttgart', 'buettstedt', 'trittau', 'gusterath', 'uttenreuth', 'baustert', 'buttstaedt', 'troestau', 'rastatt', 'grettstadt']|
|stutgart|pca_2g|['stuttgart', 'putgarten', 'holtgast', 'hallgarten', 'wustermark', 'stkatharinen', 'daetgen', 'mauth', 'garsaminn', 'marth']|
|stutgart|pca_12g|['stuttgart', 'putgarten', 'gusterath', 'rastatt', 'ottrau', 'abstatt', 'hattert', 'klettgau', 'astert', 'lautert']|
|stutgart|som_1g|['gruenstadt', 'guenstedt', 'gusterath', 'pfungstadt', 'burgkunstadt', 'buetthard', 'straubenhardt', 'freyburgunstrut', 'luetetsburg', 'tautenburg']|
|stutgart|som_2g|['stuttgart', 'tutow', 'alttucheband', 'altusried', 'stutensee', 'landstuhl', 'stuhr', 'tutzing', 'stuvenborn', 'tuchenbach']|

As you can see, the 1grams look very similar by all the nearest and the pca method, they also appear to be the most usable. The som
method is a bit more off - not complete nonsense, but off.

For 2grams work also good for the nearest and the pca method, depending on how broken the input is.

Quite interesting are also the result when n_gram is a range (1 - 2)

#TODOs
* consider range for n-grams - see CountVectorizer doc
* try other similarity method
* compare with http://www.norvig.com/spell-correct.html
