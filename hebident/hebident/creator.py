# builtins
import os
import shutil
import datetime
import random
import tempfile
from collections import defaultdict
from typing import Sequence, Optional

# dependencies
import dill
import torch
import numpy as np
import pandas as pd

# local
from .core import HebelomaTrainer, ClassifierEx, SubClassifier
from .torch_wrap import TorchClassifier
from .metrics import f1m_score_for_class
from .util import DurationEstimator, nothing_printer


def _set_seeds():
    """
        Random numbers are used at various points in the identifier
        E.g. for dividing collections into a testing and training set in a random fashion
        For reproducibility is important to set the seed consistently, which is done here
    Returns:
        None
    """
    super_seed = 111    # Feel free to change this one
    # Let these be changed as a knock-on consequence
    random.seed(super_seed + 137)
    torch.manual_seed(super_seed)
    np.random.seed(super_seed + 80114)


def _dill_save(data, filename):
    dirname = os.path.dirname(filename)
    os.makedirs(dirname, exist_ok=True)
    with open(filename, "wb") as f:
        dill.dump(data, f)


class _Metrics:

    def __init__(self, pos_array, average_prob, mrr, f1):
        # Old stuff
        # metrics = (out_sample_pos[0],
        #           out_sample_pos[0] + out_sample_pos[1],
        #           out_sample_pos[0] + out_sample_pos[1] + out_sample_pos[2],
        #           out_sample_pos[0] + out_sample_pos[1] + out_sample_pos[2] + out_sample_pos[3],
        #           out_sample_pos[0] + out_sample_pos[1] + out_sample_pos[2] + out_sample_pos[3] + out_sample_pos[4],
        #           num_testing_samples,
        #           100 * (out_sample_pos[0] + out_sample_pos[1] + out_sample_pos[2]) / num_testing_samples,
        #           average_prob)
        self.top1_count = pos_array[0]                    # index 0 in old
        self.top2_count = self.top1_count + pos_array[1]  # index 1 in old
        self.top3_count = self.top2_count + pos_array[2]  # index 2 in old
        self.top4_count = self.top3_count + pos_array[3]  # index 3 in old
        self.top5_count = self.top4_count + pos_array[4]  # index 4 in old
        self.total_samples = sum(pos_array)               # index 5
        self.top3_percent = 100.0*self.top3_count / self.total_samples  # index 6
        self.average_prob = average_prob                  # index 7
        self.mrr = mrr                                    # Not in old
        self.f1 = f1                                      # Not in old
        self.learning_rate = None
        self.filename = None
        self.max_epochs = None

    def repr(self):
        # print(f"Classifier stage {ix}: SP: top 1: {metrics.} top 3: {metrics[2]} top 5: {metrics[4]}  total: {metrics[5]} (top 3 %age: {metrics[6]}, avg prob {metrics[7]})")
        s = f"top 1: {self.top1_count}"
        s += f" top 3: {self.top3_count}"
        s += f" top 5: {self.top5_count}"
        s += f" Total: {self.total_samples}"
        s += f" %age: {self.top3_percent}"
        s += f" avg prob : {self.average_prob}"
        s += f" mrr : {int(100*self.mrr)}"
        s += f" f1 : {int(100*self.f1)}"
        return s

    def final_line(self):
        t1 = 100.0*self.top1_count/self.total_samples
        t3 = 100.0*self.top3_count/self.total_samples
        t5 = 100.0*self.top5_count/self.total_samples
        return f"{self.max_epochs},{self.learning_rate},{self.total_samples},{t1},{t3},{t5},{100.0*self.mrr},{100.0*self.f1}"


def _check_if_best(curr_best: Optional[_Metrics], this_learning_rate, this_max_epochs, this_metrics: _Metrics, this_filename):
    """
    For a given set of parameters (e.g. choice of optimizer type), a given learning rate and number of epochs
    will give the best result (as measured by MRR). This function keeps track of which learning rate and epochs was best,
    and the dill file representing that identifier
    """
    is_new_best = False
    if curr_best is None:
        curr_best = this_metrics
        is_new_best = True
    else:
        # Use mrr as determiner of "best"
        if this_metrics.mrr > curr_best.mrr:
            is_new_best = True

    if is_new_best:
        curr_best.learning_rate = this_learning_rate
        curr_best.filename = this_filename
        curr_best.max_epochs = this_max_epochs

    return curr_best


def _create_classifiable_data(parsers_names: Sequence[str],
                              collections_df,
                              training_set_proportion,
                              missing_allowance,
                              require_full_testing_data,
                              include_continent_pre_filter):
    trainer = HebelomaTrainer(parsers_names,
                              training_set_proportion,
                              missing_allowance,
                              require_full_testing_data,
                              include_continent_pre_filter)
    training_df, testing_df = trainer.train(collections_df=collections_df)
    return training_df, testing_df, trainer


def _create_pytorch_subclassifier(classifier_base,
                                  classifier_ix,
                                  training_df,
                                  trainer: HebelomaTrainer,
                                  target_type: str,
                                  redo_threshold_for_secondary_classifiers,
                                  print_function):

    assert target_type in trainer.target_columns(), f"Target type is {target_type}"

    relevant_row_ixs = trainer.training_set_row_indexes_for(classifier_ix)
    relevant_rows = training_df.iloc[relevant_row_ixs]
    training_df_training_columns = relevant_rows[trainer.regression_columns(classifier_ix)]

    print_function(f"About to create subclassifier {classifier_ix}. There are {len(relevant_row_ixs)} applicable training rows.")

    classifier_base.fit(training_df=training_df_training_columns,
                        training_target=relevant_rows[target_type],
                        print_function=print_function)
    parser_name = trainer.parsers_names[classifier_ix]
    return SubClassifier(parser_name, classifier_base, trainer.filter_columns(classifier_ix), redo_threshold_for_secondary_classifiers)


class _ClassifierCreator:

    def __init__(self,
                 collections_df,
                 parsers_set: Sequence[str],
                 include_continent_pre_filter,
                 training_set_proportion,
                 missing_allowance,
                 require_full_testing_data):

        self.training_data, self.testing_data, self.trainer = _create_classifiable_data(parsers_set,
                                                                                        collections_df,
                                                                                        training_set_proportion,
                                                                                        missing_allowance,
                                                                                        require_full_testing_data,
                                                                                        include_continent_pre_filter)
        # print(self.training_data)
        # print("=== End of training / start of testing ===")
        # print(self.testing_data)
        self.parsers_set = parsers_set
        self.num_classifiers = len(parsers_set)  # noqa, for now
        self.missing_allowance = missing_allowance
        self.require_full_testing_data = require_full_testing_data
        self.continent_pre_filter = include_continent_pre_filter

    def _do_testing_and_save(self, clex, intermediate_output_directory, grouping_type):
        # The same for sklearn and pytorch - it has been abstracted away
        class_is_species = grouping_type.lower() == "species"
        # We don't use the result, but this is not a redundant call
        # it has a side effect of saving down the training set results
        _ = clex.classify_from_df(self.training_data,  self.trainer.training_set_ids, "training", class_is_species)
        out_of_sample_results = clex.classify_from_df(self.testing_data, self.trainer.testing_set_ids, "testing", class_is_species)

        # Only now that we have also saved down for the full classification
        # results (for future display) do we save the classifier
        self.today_str = datetime.datetime.now().strftime("%Y%m%d")
        filename = f"classifier_{clex.context}_{self.today_str}.dill"
        temporary_storage_filename = os.path.join(intermediate_output_directory, filename)
        _dill_save(clex, temporary_storage_filename)

        all_metrics = []
        full_dict = dict()
        # Here we are iterating over the all sub-classifiers
        # I.e. we also show results for the intermediate classifier
        for ixx, oos in enumerate(out_of_sample_results):
            is_last = ixx == len(out_of_sample_results) - 1
            out_sample_pos = oos["positions_array"]
            by_id_data_dict = oos["by_id"]
            prob_sum = 0.0
            mrr_sum = 0.0

            def three_zeroes():
                return [0, 0, 0]  # true positive, false negative, false positive

            f1_score_dict = defaultdict(three_zeroes)

            for idd, data in by_id_data_dict.items():
                # data here is (guess_ix, real, res, guess_prob)
                # res is key because this is the full ordering of name, prob pairs
                # guess_ix gives an MRR score via 1/(1+guess_ix)
                # true positive  <===> guess_ix == 0
                # false negative <===> guess_ix != 1
                # false positive -> goes to the species we did assign most prob too
                if is_last:
                    full_dict[idd] = data

                guess_ix = data[0]
                real_name = data[1]
                guessed_name = data[2][0][0]
                prob_of_correct = data[3]
                prob_sum += prob_of_correct

                mrr_sum += 1.0/(1+guess_ix)
                if guess_ix == 0:
                    f1_score_dict[guessed_name][0] += 1
                else:
                    f1_score_dict[real_name][1] += 1
                    f1_score_dict[guessed_name][2] += 1

            f1_score = 0.0
            for class_name, (true_pos, false_neg, false_pos) in f1_score_dict.items():
                f1_score += f1m_score_for_class(true_pos, false_pos, false_neg)
            f1_score /= len(f1_score_dict)
            mrr_sum /= len(by_id_data_dict)

            # positions array gives us the number of guesses that
            # were in that position, so index 0 is the number of guesses that were right
            num_testing_samples = sum(out_sample_pos)
            average_prob = prob_sum / num_testing_samples

            metrics = _Metrics(out_sample_pos, average_prob, mrr_sum, f1_score)
            all_metrics.append(metrics)
        return all_metrics, temporary_storage_filename, full_dict

    def fit_one_pytorch_one_grouping_type(self,
                                          print_function,
                                          intermediate_output_directory,
                                          grouping_type,
                                          optimizer_type,
                                          amsgrad,
                                          max_epochs,
                                          learning_rates,
                                          layer_definitions,
                                          redo_threshold_for_secondary_classifiers):
        subclassifiers = []
        lr_str = "_".join([str(lr) for lr in learning_rates]).replace(".", "")
        context = f"pytorch_{grouping_type}_{lr_str}"
        for classifier_ix in range(self.num_classifiers):
            classifier_base = TorchClassifier(optimizer_type=optimizer_type,
                                              amsgrad=amsgrad,
                                              max_epochs=max_epochs[classifier_ix],
                                              learning_rate=learning_rates[classifier_ix],
                                              layer_definition=layer_definitions[classifier_ix])
            this_classifier = _create_pytorch_subclassifier(classifier_base,
                                                            classifier_ix,
                                                            self.training_data,
                                                            self.trainer,
                                                            grouping_type,
                                                            redo_threshold_for_secondary_classifiers,
                                                            print_function)
            subclassifiers.append(this_classifier)
        clex = ClassifierEx(subclassifiers, self.trainer, context, target_type=grouping_type)
        print_function(f"Created fit for {context}")
        return self._do_testing_and_save(clex, intermediate_output_directory, grouping_type)


def _learning_rate_to_max_epochs(lrs, optimizer_type):

    def _lr_to_max_epoch_one(lr):
        if optimizer_type in ("adam", "adamw"):
            # 0.001  -> 100
            # 0.0001 -> 1000
            return max(100, int(1/(lr*10)))
        elif optimizer_type in ("sgd", ):
            return 1000

    return [_lr_to_max_epoch_one(x) for x in lrs]


def _create_and_test_identifier_pair(intermediate_output_directory,
                                     final_output_directory,
                                     printable_characters_def,
                                     allow_missing_data,
                                     cc_species: _ClassifierCreator,
                                     cc_section: _ClassifierCreator,
                                     optimizer_type,
                                     amsgrad,
                                     layer_definition,
                                     redo_threshold_for_secondary_classifiers,
                                     verbose_print_function,
                                     laconic_print_function):

    if optimizer_type in ("adam", "adamw"):
        learning_rates_to_try = 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05
    elif optimizer_type in ("sgd", ):
        learning_rates_to_try = 0.2, 0.1, 0.05
    else:
        raise RuntimeError("Unknown optimizer_type")
    best_section = None
    best_species = None
    sp_layer_definitions = [layer_definition]*cc_species.num_classifiers
    st_layer_definitions = [layer_definition]*cc_section.num_classifiers

    for lr in learning_rates_to_try:
        sp_learning_rates = [lr]*cc_species.num_classifiers
        sp_max_epochs = _learning_rate_to_max_epochs(sp_learning_rates, optimizer_type)
        st_learning_rates = [lr]*cc_section.num_classifiers
        st_max_epochs = _learning_rate_to_max_epochs(st_learning_rates, optimizer_type)
        verbose_print_function("==================")
        verbose_print_function(f"Max epochs (species)     : {sp_max_epochs}")
        verbose_print_function(f"Learning rates (species) : {sp_learning_rates}")
        verbose_print_function(f"Max epochs (section)     : {st_max_epochs}")
        verbose_print_function(f"Learning rates (section) : {st_learning_rates}")

        _set_seeds()  # Critical for reproducibility

        sp_metrics, last_species_filename, sp_full_dict = cc_species.fit_one_pytorch_one_grouping_type(
            print_function=verbose_print_function,
            intermediate_output_directory=intermediate_output_directory,
            grouping_type="Species",
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            max_epochs=sp_max_epochs,
            learning_rates=sp_learning_rates,
            layer_definitions=sp_layer_definitions,
            redo_threshold_for_secondary_classifiers=redo_threshold_for_secondary_classifiers)

        for ix, metrics in enumerate(sp_metrics):
            verbose_print_function(f"Classifier stage {ix}: SP: {metrics.repr()}")

        best_species = _check_if_best(best_species, sp_learning_rates, sp_max_epochs, sp_metrics[-1], last_species_filename)

        # Critical for reproducibility - and also forces section and species
        # to use same training set
        _set_seeds()

        st_metrics, last_section_filename, st_full_dict = cc_section.fit_one_pytorch_one_grouping_type(
            print_function=verbose_print_function,
            intermediate_output_directory=intermediate_output_directory,
            grouping_type="Section",
            optimizer_type=optimizer_type,
            amsgrad=amsgrad,
            max_epochs=st_max_epochs,
            learning_rates=st_learning_rates,
            layer_definitions=st_layer_definitions,
            redo_threshold_for_secondary_classifiers=redo_threshold_for_secondary_classifiers)

        for ix, metrics in enumerate(st_metrics):
            verbose_print_function(f"Classifier stage {ix}: ST: {metrics.repr()}")

        best_section = _check_if_best(best_section, st_learning_rates, st_max_epochs, st_metrics[-1], last_section_filename)

    # SAVE THE BEST ONE
    # What does saving mean?
    # Actually means copying from the already-saved location to the
    # real static location

    verbose_print_function(f"Overall best learning rate for species was {best_species.learning_rate}")
    verbose_print_function(f"Overall best learning rate for section was {best_section.learning_rate}")

    layer_definition_str = "layer_def_" + "_".join(layer_definition)

    # ### REALLY PRINT THE FINAL LINE
    # Class group | Character set | Allow one missing element | Minimiser
    laconic_print_function(f"Species,{printable_characters_def},{allow_missing_data},{cc_species.continent_pre_filter},{optimizer_type},{amsgrad},{layer_definition_str},{best_species.final_line()}")
    laconic_print_function(f"Section,{printable_characters_def},{allow_missing_data},{cc_section.continent_pre_filter},{optimizer_type},{amsgrad},{layer_definition_str},{best_section.final_line()}")

    sp_filename = f"species_{printable_characters_def}_md{allow_missing_data}_contpf{cc_species.continent_pre_filter}_{optimizer_type}_amsgrad{amsgrad}_{layer_definition_str}.dill"
    st_filename = f"section_{printable_characters_def}_md{allow_missing_data}_contpf{cc_section.continent_pre_filter}_{optimizer_type}_amsgrad{amsgrad}_{layer_definition_str}.dill"

    sp_dst = os.path.join(final_output_directory, sp_filename)
    if os.path.exists(sp_dst):
        laconic_print_function(f"Warning: Overwriting existing species identifier dill file {sp_dst}")
    shutil.copyfile(best_species.filename, sp_dst)

    st_dst = os.path.join(final_output_directory, st_filename)
    if os.path.exists(st_dst):
        laconic_print_function(f"Warning: Overwriting existing section identifier dill file {st_dst}")
    shutil.copyfile(best_section.filename, st_dst)

    laconic_print_function(f"Species and section identifiers saved to {final_output_directory}")


def create_classifiers(collections_df: pd.DataFrame,
                       output_directory: str,
                       logger):
    """
    Args:
        collections_df: This is a dataframe where the rows are collections and the columns are features
                        The values are presumed to be **transformed** form - i.e. a value between 0 and 1
                        *not* an original value
        output_directory: This directory (created if necessary) will contain two identifiers
                          for each 'experiment' i.e. each choice of parameters that is tried
                          - one identifier identifies to species
                          - the other identifier identifies to section

    Returns:
    """
    os.makedirs(output_directory, exist_ok=True)
    laconic_print_function = logger
    verbose_print_function = nothing_printer

    experiments = []

    all_continent_filters = False, True
    all_optimisers = "adamw", "adam", "sgd"
    all_amsgrads = True, False
    all_layer_defs = [["relu", "relu"], ["mish"], ["relu", "mish"], ["mish", "relu"],  ["mish", "mish"], ["relu"]]
    all_missing_allowances = 1, 0

    primary_character_groups = "CG1", "CG2", "CG3", "CG4", "CG5", "CG6", "CG7", "CG8", "CG9"
    all_amyg_separate = None, "CG1", "CG2", "CG3", "CG4", "CG5", "CG6", "CG7", "CG8", "CG9", "CG10", "CG11"

    redo_threshold_for_secondary_classifiers = 0.9
    training_set_proportion = 0.7
    require_full_testing_data = True

    def contains_at_most_one_non_primary(this_pcg, this_amy_separate, this_optimiser, this_amsgrad, this_layer_def, this_continent_filter, this_ma):
        primaries = [this_pcg in ("CG7", "CG8"),
                     this_amy_separate is None,
                     this_optimiser == "adamw",
                     this_amsgrad,
                     this_layer_def == ["relu"],
                     not this_continent_filter,
                     this_ma == 1]
        primary_count = sum(1 if x else 0 for x in primaries)
        return primary_count >= (len(primaries) - 1)

    # Parser set, optimizer type, amsgrad, layers def, include continent prefilter
    for cf in all_continent_filters:        # 2x3x2x9x12x5x2x2  ~~ 2,500
        for optimiser in all_optimisers:
            for amsgrad in all_amsgrads:
                for amyg_separate in all_amyg_separate:
                    for primary_character_group in primary_character_groups:
                        for layer_def in all_layer_defs:
                            for missing_allowance in all_missing_allowances:
                                if contains_at_most_one_non_primary(primary_character_group, amyg_separate, optimiser, amsgrad, layer_def, cf, missing_allowance):
                                    this_experiment = primary_character_group, amyg_separate, optimiser, amsgrad, layer_def, cf, missing_allowance
                                    experiments.append(this_experiment)

    num_experiments = len(experiments)
    laconic_print_function(f"Number of experiments = {num_experiments}")
    de = DurationEstimator(num_experiments)

    with tempfile.TemporaryDirectory() as intermediate_dir:
        for cnt, experiment in enumerate(experiments):
            primary_character_group, amyg_separate, optimizer_type, amsgrad, layer_definition, include_continent_pre_filter, missing_allowance = experiment

            # We end up with section_characters like ("CG1",)
            # And species characters like ("CG1",) or ("CG1, amygdalina_CG8")
            section_characters = primary_character_group,
            if amyg_separate is not None:
                species_characters = (primary_character_group, f"amygdalina_{amyg_separate}")
            else:
                species_characters = primary_character_group,

            _set_seeds()
            cc_species = _ClassifierCreator(collections_df=collections_df,
                                            parsers_set=species_characters,
                                            include_continent_pre_filter=include_continent_pre_filter,
                                            training_set_proportion=training_set_proportion,
                                            missing_allowance=missing_allowance,
                                            require_full_testing_data=require_full_testing_data)

            _set_seeds()  # Need to do this to ensure same training and testing set for species and sect
            cc_section = _ClassifierCreator(collections_df=collections_df,
                                            parsers_set=section_characters,
                                            include_continent_pre_filter=include_continent_pre_filter,
                                            training_set_proportion=training_set_proportion,
                                            missing_allowance=missing_allowance,
                                            require_full_testing_data=require_full_testing_data)

            printable_characters_def = "_".join(species_characters)

            _create_and_test_identifier_pair(intermediate_dir,
                                             output_directory,
                                             printable_characters_def,
                                             missing_allowance == 1,
                                             cc_species,
                                             cc_section,
                                             optimizer_type,
                                             amsgrad,
                                             layer_definition,
                                             redo_threshold_for_secondary_classifiers,
                                             verbose_print_function,
                                             laconic_print_function)
            de.print_estimate(cnt+1)
