import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for n in range(test_set.num_items):
        prob = float("-inf")
        logL = dict()
        guess = ''
        x, lengths = test_set.get_item_Xlengths(n)
        for word, model in models.items():
            try:
                s = model.score(x, lengths)
                logL[word] = s

                if s > prob:
                    prob = s
                    guess = word

            except:
                logL[word] = float("-inf")
                pass
        probabilities.append(logL)
        guesses.append(guess.rstrip('123456789'))

    return probabilities, guesses