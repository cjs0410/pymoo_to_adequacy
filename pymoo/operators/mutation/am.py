import numpy as np

from pymoo.core.mutation import Mutation


class AdequacyMutation(Mutation):
    def __init__(self, prob=1.0, prob_var=None, adeq_scores=None, **kwargs) -> None:
        super().__init__(prob, prob_var, **kwargs)
        self.adeq_scores = adeq_scores

    def _do(self, problem, X, **kwargs):
        Xp = np.copy(X)
        
        prob_var = np.zeros(X.shape)
        for i, row in enumerate(X):
            for j in range(len(row)):
                if row[j]:
                    prob_var[i][j] = self.adeq_scores[0][j]
                else:
                    prob_var[i][j] = self.adeq_scores[1][j]
                    
        flip = np.random.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp


class AM(AdequacyMutation):
    pass
