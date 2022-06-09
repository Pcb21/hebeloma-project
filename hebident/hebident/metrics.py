import math
import scipy.optimize


class C2Finder:

    def __init__(self, probs, alpha):
        # log(P(i)) = S_i - log(Sum_i(exp(S_i)))  (1)
        # log(P'(i)) = alpha*S_i - log(Sum_i(exp(alpha S_i)))
        # Let C = 1
        self.alpha = alpha
        c1 = 1.0

        # To avoid numerical issues we do NOT solve those with really tiny probs - leave them alone
        alone_threshold = 1e-8
        self.original_probs = probs
        self.sum_of_smalls = 0.0
        self.small_indexes = []
        self.large_indexes = []
        large_probs = []
        for ix, p in enumerate(probs):
            if p < alone_threshold:
                self.small_indexes.append(ix)
                self.sum_of_smalls += p
            else:
                self.large_indexes.append(ix)
                large_probs.append(p)
        try:
            log_probs = [math.log(p) for p in large_probs]
            scores = [log_p + c1 for log_p in log_probs]
            self.alpha_scores = [alpha * S_i for S_i in scores]
        except ValueError:
            raise RuntimeError(f"Got domain error trying to probs {probs} and alpha {alpha}")

    @staticmethod
    def starting_point():
        return 1.0

    def _alt_probs_impl(self, c2):
        return [math.exp(alphaS_i - c2) for alphaS_i in self.alpha_scores]

    def final_alt_probs(self, c2):
        # print(f"alpha was {self.alpha}: Final C2 was: {c2}")
        alt_large_probs = self._alt_probs_impl(c2)
        if not self.small_indexes:
            return alt_large_probs  # no small probs to re-integrate back in

        sum_large = sum(alt_large_probs)
        target = 1.0 - self.sum_of_smalls
        ratio = target/sum_large
        alt_large_probs = [a*ratio for a in alt_large_probs]
        result = [0.0]*(len(self.small_indexes) + len(self.large_indexes))
        for ix in self.small_indexes:
            result[ix] = self.original_probs[ix]  # unchanged
        for counting_ix, output_ix in enumerate(self.large_indexes):
            result[output_ix] = alt_large_probs[counting_ix]
        return result

    def __call__(self, c2):
        sum_alt_probs = sum(self._alt_probs_impl(c2))
        res = sum_alt_probs - 1.0
        # derivative = Sum_i derivative(exp(alphaSi - C2))
        # Sum_i - exp(alphaS_i -C 2)
        # Return value, derivative as a pair, and set fprime=True in the root finder
        return res, -1.0 * sum_alt_probs


def invert_softmax(probs, alpha):
    func = C2Finder(probs, alpha)
    res = scipy.optimize.root_scalar(func, x0=func.starting_point(), method="newton", fprime=True)
    if not res.converged:
        raise RuntimeError("Failed to find a solution for alpha={alpha}")
    return func.final_alt_probs(res.root)


def f1m_score_for_class(true_positives, false_positives, false_negatives):
    # https://vitalflux.com/micro-average-macro-average-scoring-metrics-multi-class-classification-python/
    # From https://arxiv.org/pdf/2103.10107.pdf
    # tp = true positive in that class
    # fp = false positive in that class (i.e. predicted that class, but was something else)
    # fn = false negative in that class (i.e. was that class, but predicted something else)
    tp = true_positives
    fp = false_positives
    fn = false_negatives
    if tp == 0:
        return 0
        # raise RuntimeError(f"Bad data for class '{class_name}' ({tp}, {fp}, {fn})")

    ps = tp/(tp + fp)
    rs = tp/(tp + fn)
    fs = 2*(rs * ps)/(rs + ps)
    return fs
