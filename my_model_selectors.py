import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        N = len(self.lengths)
        logN = np.log(N)

        bic = float("inf")
        ret = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)

                # |transition matrix - 1 row| + free starting probabilities + #means (|covars matrix|)
                # = N(N-1) + (N-1) + N*n
                # = N^2 - N + N - 1 + 2(N*n)
                # = N^2 + N*n -1
                p = N ** 2 + 2 * n * N - 1

                logL = model.score(self.X, self.lengths)

                new_bic = -2 * logL + p * logN

                if new_bic < bic:
                    bic = new_bic
                    ret = model

            except:
                continue

        return ret


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        """
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        dic = float("-inf")
        ret = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = self.base_model(n)
                logL = model.score(self.X, self.lengths)

                logP = []
                for word, (X, lengths) in self.hwords.items():
                    if word != self.this_word:
                        logP.append(model.score(X, lengths))

                new_dic = logL - sum(logP) / (len(logP) - 1)

                if new_dic > dic:
                    dic = new_dic
                    ret = model

            except:
                continue

        return ret


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        cv = float("-inf")
        ret = None

        for n in range(self.min_n_components, self.max_n_components + 1):
            try:
                # leave-one-out cross-validation limited at 50 folds for performance
                n_splits = min(len(self.sequences), 20)

                kf = KFold(n_splits=n_splits, shuffle=True)

                scores = []
                for train, test in kf.split(self.sequences):
                    try:
                        X_train, lengths_train = combine_sequences(train, self.sequences)
                        model = GaussianHMM(n_components=n, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)

                        X_test, lengths_test = combine_sequences(test, self.sequences)
    
                        scores.append(model.score(X_test, lengths_test))
                    except:
                        continue

                new_cv = np.mean(scores)

                if new_cv > cv:
                    cv = new_cv
                    ret = model

            except:
                continue

        return ret