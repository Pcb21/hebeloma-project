import dill
from collections import defaultdict
from .metrics import f1m_score_for_class
from .core import species_generalised_section


def _accumulate_by_id_data_section(by_id_data):

    def five_zeroes():
        # mrr_score, total in this class, true pos in this class, false neg, false pos
        return [0, 0, 0, 0, 0]

    by_class_temp = defaultdict(five_zeroes)

    for specimen_id, value in by_id_data.items():
        # pos_ix -
        pos_ix = value[0]
        is_true_positive = (pos_ix == 0)
        actual_name = value[1]
        probs = value[2]
        guessed_name = probs[0][0]   # Probs are in order and a 2-tuple name first
        guess_mrr_score = 1.0/float(1+pos_ix)

        by_class_temp[actual_name][0] += guess_mrr_score
        by_class_temp[actual_name][1] += 1

        if is_true_positive:
            by_class_temp[actual_name][2] += 1
        else:
            by_class_temp[actual_name][3] += 1
            by_class_temp[guessed_name][4] += 1

    by_class_extra_data = dict()
    for section, mrr_data in by_class_temp.items():
        mrr_sum, mrr_count, true_pos, false_neg, false_pos = mrr_data
        mrr_score = mrr_sum/float(mrr_count)
        f1_score = f1m_score_for_class(true_pos, false_neg, false_pos)
        by_class_extra_data[section] = {"MRRScore": mrr_score,
                                        "F1Score": f1_score}
    return by_class_extra_data


def _accumulate_by_id_data(by_id_data):
    """
    The saved down data in the DILL files has a lot of information (a full 'guesses_array')
    for each collection we tried to classify in the training set

    This consumes a lot of memory and can fail in limited memory environment such as AWS - so thin it down here!
    """
    # results is a dict of (id: pos_ix, real_name, guesses_array)
    results = by_id_data
    wrong_ones_data = []

    probs_of_correct = []
    probs_of_incorrect = []

    def five_zeroes():
        return [0, 0, 0, 0, 0]

    by_class_temp = defaultdict(five_zeroes)

    for specimen_id, value in results.items():
        # The value in results might be a 3-tuple (if the classifier was created before we started
        # storing the probability of the right species as a fourth element) or a 4-tuple
        # otherwise
        # The code here is safe in the face of both formulations
        pos_ix = value[0]
        actual_name = value[1]
        guesses_array = value[2]
        is_true_positive = (pos_ix == 0)

        classifier_top_prob = guesses_array[0][1]
        guessed_name = guesses_array[0][0]
        classifier_species_probability = guesses_array[pos_ix][1]
        assert(isinstance(classifier_species_probability, float))

        guess_rank = pos_ix + 1
        guess_mrr_score = 1.0/float(guess_rank)
        by_class_temp[actual_name][0] += guess_mrr_score
        by_class_temp[actual_name][1] += 1

        if is_true_positive:
            by_class_temp[actual_name][2] += 1
        else:
            by_class_temp[actual_name][3] += 1
            by_class_temp[guessed_name][4] += 1

        if is_true_positive:
            probs_of_correct.append((specimen_id, classifier_species_probability))
            # Here is where you can hack it if you need to find an ID that the identifier worked for
            # if actual_name == "eburneum":
            #   print(f"eburneum correct! {specimen_id}")
            continue  # Ignore the ones we got right

        probs_of_incorrect.append((specimen_id, classifier_species_probability, classifier_top_prob))
        top_guess = guesses_array[0][0]
        actual_section = species_generalised_section(actual_name)
        top_guess_section = species_generalised_section(top_guess)
        section_correct = "YES" if actual_section == top_guess_section else "NO"
        classifier_top_prob_str = f"{100.0*classifier_top_prob:.2f}"
        classifier_species_prob_str = f"{100.0*classifier_species_probability:.2f}"
        wrong_ones_data.append([specimen_id, actual_name, top_guess, guess_rank, classifier_top_prob_str, classifier_species_prob_str, actual_section, top_guess_section, section_correct])
    wrong_ones_data.sort(key=lambda x: x[6])
    wrong_ones_data.sort(key=lambda x: x[0])  # Sort second by ID

    probs_of_correct.sort(key=lambda x: x[1], reverse=True)
    probs_of_incorrect.sort(key=lambda x: x[1], reverse=True)

    scatter_data = [{"x": species_prob, "y": top_prob} for _, species_prob, top_prob in probs_of_incorrect]
    scatter_collections = [specimen_number for specimen_number, _, _ in probs_of_incorrect]
    scatter_dict = {"data": scatter_data,
                    "collections": scatter_collections,
                    "fit": {"type": "none"}}

    probs_of_incorrect = [(x[0], x[1]) for x in probs_of_incorrect]

    by_class_extra_data = dict()
    for species, mrr_data in by_class_temp.items():
        mrr_sum, mrr_count, true_pos, false_neg, false_pos = mrr_data
        mrr_score = 0 if mrr_count == 0 else mrr_sum/float(mrr_count)
        f1_score = f1m_score_for_class(true_pos, false_pos, false_neg)
        by_class_extra_data[species] = {"MRRScore": mrr_score,
                                        "F1Score": f1_score}

    return wrong_ones_data, probs_of_correct, probs_of_incorrect, scatter_dict, by_class_extra_data


class Loader(dill.Unpickler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .torch_wrap import TorchClassifier  # noqa

    def find_class(self, module, name):
        # Backwards compatibility for old structure of files
        if module == "hebeloma.torch_wrap":
            module = "hebident.torch_wrap"
        if module == "hebeloma.views_identifier":
            module = "hebident.core"
        if module == "hebeloma.views_utils" and name in ("parse_lat", "parse_lng"):
            module = "hebident.geography"
        if module == "hebeloma.parse_utils" and name in ("SToSingle",):
            module = "hebident.parse"
        if module.startswith("hebeloma"):
            raise RuntimeError(f"Oh dear ... still looking for {module}::{name}")
        return super().find_class(module, name)


def load_thinned_identifier(filename, is_species):

    with open(filename, "rb") as f:
        data = Loader(f).load()

    # To save memory, throw away all but the results of the last classifier
    # and thing down the "by_id" data
    for key in data.all_classification_results:
        if isinstance(data.all_classification_results[key], list):
            data.all_classification_results[key] = data.all_classification_results[key][-1]
        id_data = data.all_classification_results[key]["by_id"]
        if is_species:
            thinned_by_id_data, probs_of_correct, probs_of_incorrect, scatter_dict, by_class_extra_data = _accumulate_by_id_data(id_data)
            data.all_classification_results[key]["wrong_guess_data"] = thinned_by_id_data
            data.all_classification_results[key]["probs_of_correct"] = probs_of_correct
            data.all_classification_results[key]["probs_of_incorrect"] = probs_of_incorrect
            data.all_classification_results[key]["incorrect_vs_top_scatter"] = scatter_dict
            data.all_classification_results[key]["by_class_extra_data"] = by_class_extra_data
        else:
            by_class_extra_data = _accumulate_by_id_data_section(id_data)
            data.all_classification_results[key]["by_class_extra_data"] = by_class_extra_data
        data.all_classification_results[key]["by_id"] = None  # reclaim the memory
    return data
