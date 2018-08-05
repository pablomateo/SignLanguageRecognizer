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
        raise NotImplementedError

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
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        tgtScore, tgtModel = float("inf"), None

        for components in range(self.min_n_components, self.max_n_components + 1):
            try:
                chModel = self.base_model(components)
                chModelLen = chModel.score(self.X, self.lengths)

                numCharct = self.X.shape[1]
                numVrs = components * (components - 1) + 2 * numCharct * components

                lgNum = np.log(self.X.shape[0])
                bicResult = -2 * chModelLen + numVrs * lgNum

                if bicResult < tgtScore:
                    tgtScore, tgtModel = bicResult, chModel

            except Exception as _:
                continue

        return tgtModel if tgtModel is not None else self.base_model(self.n_constant)



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''


    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        palabrsNvs = []
        modelos = []
        puntAcumul = []

        for palabra in self.words:
            if palabra != self.this_word:
                palabrsNvs.append(self.hwords[palabra])

        try:
            for stateNm in range(self.min_n_components, self.max_n_components + 1):
                markvModel = self.base_model(stateNm)

                topWrd = markvModel.score(self.X, self.lengths)
                modelos.append((topWrd, markvModel))

        except Exception as e:
            pass

        for indicePos, model in enumerate(modelos):
            topWrd, markvModel = model

            wordPos = [model[1].score(palabr[0], palabr[1]) for palabr in palabrsNvs]

            score_dic = topWrd - np.mean(wordPos)
            puntAcumul.append(tuple([score_dic, model[1]]))

            compMax = max(puntAcumul, key = lambda x: x[0])
        return compMax[1] if puntAcumul else None




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        mediasArray = []
        foldSequences = KFold()

        try:
            for componentN in self.n_components:
                compBaseModel = self.base_model(componentN)
                foldValues = []

                for _, seqObs in foldSequences.split(self.sequences):
                    seqObsX, seqObsL = combine_sequences(seqObs, self.sequences)
                    foldValues.append(compBaseModel.score(seqObsX, seqObsL))

                mediasArray.append(np.mean(foldValues))

        except Exception as _:
            pass

        retStates = self.n_components[np.argmax(mediasArray)] if mediasArray else self.n_constant
        return self.base_model(retStates)


