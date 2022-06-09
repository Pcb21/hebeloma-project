# builtin modules
import abc
import random
import math
import bisect
import os
import csv
from typing import Sequence, Iterable
from collections import defaultdict

# additional modules
import pandas as pd

# local
from .parse import SToSingle


COLL_ID_NOT_PRESENT = 56
COLL_ID_IN_TRAINING_SET = 57
COLL_ID_NOT_IN_TRAINING_SET = 58
species_generalised_section_data = {}


def species_generalised_section(sp):
    global species_generalised_section_data
    if not species_generalised_section_data:
        with open(os.path.join(os.path.dirname(__file__), "sgs.csv"), "r") as csvfile:
            r = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in r:
                name, gs, continents = row
                species_generalised_section_data[name] = gs, continents
    if sp not in species_generalised_section_data:
        raise RuntimeError(f"Can't identify section for '{sp}'")
    return species_generalised_section_data[sp][0]


class BadIdentifierValue(RuntimeError):
    pass


class _PrintLogger:

    @staticmethod
    def write(msg):
        print(msg)


class _BlankLogger:

    @staticmethod
    def write(msg):
        pass


class _NotPresentTag:
    pass


_NotPresent = _NotPresentTag()


class CFieldHelper:

    def __init__(self, primary_names, display_names, mapping_dict, request_name):
        self.primary_names = primary_names
        self.display_names = display_names
        self.mapping_dict = mapping_dict
        self.request_name = request_name

    def _make_false_array(self):
        # If I remember rightly, a simple [False]*len(self.primary_names) had referencing problems
        sz = len(self.primary_names)
        out = []
        for i in range(sz):
            out.append(False)
        return out

    def _parse_core(self, ts):
        # Given an semi-colon input string of features, returns a True/False
        # string where the input string has properties
        elems = ts.split(";")
        elems = [e.replace(' (?)', '').replace(' ', '').replace('-', "").strip().lower() for e in elems]
        elems = [e for e in elems if len(e) > 0]
        if elems:
            out = self._make_false_array()
            for elem in elems:
                if elem in self.mapping_dict:
                    out[self.mapping_dict[elem]] = True
                # else:
                #    print(f"Property '{elem}' encountered, but this was not assigned in the dictionary")
            return True, out
        else:
            return False, self._make_false_array()

    def parse_to_one_zero_none(self, ts, empty_is_trustworthy):
        elems_present, out = self._parse_core(ts)
        if elems_present:
            return [1.0 if x else 0.0 for x in out]
        else:
            if empty_is_trustworthy:
                # de-pickling hack
                my_name = getattr(self, "request_name", "cheilocystidia shape")
                raise BadIdentifierValue(f"This identifier requires a value for {my_name}")
            else:
                return [None]*(len(out))

    def parse_to_where_true(self, ts):
        # This returns an array consisting of the elements that
        # are present in the input string
        # This is the format needed to populate a multi-select
        # combo box in the front end
        _, out = self._parse_core(ts)
        res = []
        for i, elem in enumerate(out):
            if elem:
                res.append(self.primary_names[i])
        return res

    def all_options(self):
        return zip(self.primary_names, self.display_names)


class CheiloShapes(CFieldHelper):

    def __init__(self):
        # indexes     #       0             1               2                   3               4
        primary_names = "cylindrical", "ventricose", "clavategently", "clavatestipitate", "clavateventricose"
        display_names = "Cylindrical", "Ventricose (lageniform)", "Gently clavate", "Clavate-stipitate", "Clavate-ventricose (hourglass)"
        mapping_dict = {"cylindrical": 0,
                        "ventricose": 1,
                        "lageniform": 1,
                        "clavategently": 2,
                        "clavatestipitate": 3,
                        "capitatestipitate": 3,
                        "spathulatestipitate": 3,
                        "clavateventricose": 4,
                        "hourglass": 4}
        super().__init__(primary_names, display_names, mapping_dict, request_name="Cheilocystidia shape")


class CheiloShapes2(CFieldHelper):

    def __init__(self):
        # indexes     #       0             1               2                   3               4
        primary_names = ("cylindrical",       # 0
                         "ventricose",         # 1
                         "clavategently",      # 2
                         "clavatestipitate",   # 3
                         "clavateventricose",  # 4
                         "pyriformballoon")    # 5

        display_names = ("Cylindrical",                 # 0
                         "Ventricose (or lageniform)",      # 1
                         "Gently clavate",               # 2
                         "Clavate-stipitate (or capitate-stipitate or spathulate-stipitate)",  # 3
                         "Clavate-ventricose (or hourglass)",   # 4
                         "Pyriform (or balloon-like on a string)")  # 5

        mapping_dict = {"cylindrical": 0,
                        "ventricose": 1,
                        "lageniform": 1,
                        "clavategently": 2,
                        "clavatestipitate": 3,
                        "capitatestipitate": 3,
                        "spathulatestipitate": 3,
                        "clavateventricose": 4,
                        "hourglass": 4,
                        "pyriform": 5,
                        "balloonshaped": 5}

        super().__init__(primary_names, display_names, mapping_dict, request_name="Cheilocystidia shape")


class ParserBase(abc.ABC):

    def __init__(self, db_name, request_name, column_names):
        self.db_name = db_name
        self.request_name = request_name
        self.column_names = column_names

    def train_from_coll(self, coll, verbose):
        value = getattr(coll, self.db_name, _NotPresent)
        if isinstance(value, _NotPresentTag):
            raise RuntimeError(f"{self.db_name} is not a valid database field name (type of coll: {type(coll)}")
        return self.train(value, verbose)

    def transform_from_coll(self, coll, verbose):
        value = getattr(coll, self.db_name, None)
        if verbose:
            print(f"For DB name {self.db_name}, value is {value}")
            logger = _PrintLogger()
        else:
            logger = _BlankLogger()
        ret = self.transform(value, empty_is_trustworthy=False, logger=logger)
        if verbose:
            print(f"The transformed value is {ret}")
        return ret

    def data_from_coll(self, coll):
        return self.db_name, getattr(coll, self.db_name)

    def transform_from_req(self, input_getter, logger):
        input_value = input_getter(self.request_name)
        logger.write(f"{self.request_name} received value {input_value}")
        transformed_value = self.transform(input_value, empty_is_trustworthy=True, logger=logger)
        logger.write(f"{self.request_name} transformed value to {transformed_value}")
        return transformed_value

    def coll_value(self, coll):
        return getattr(coll, self.db_name, None)

    @abc.abstractmethod
    def transform(self, x, empty_is_trustworthy, logger):
        assert False

    @abc.abstractmethod
    def train(self, x, verbose):
        assert False

    @abc.abstractmethod
    def observed_ranges(self):
        assert False


class StringContainsParser(ParserBase):

    def __init__(self, db_name, request_name, elems):
        self.elems = [e.lower() for e in elems]
        super().__init__(db_name, request_name, [request_name + e for e in elems])

    def train(self, _value, _verbose):
        pass

    def transform(self, this_string, empty_is_trustworthy, logger):
        ts = this_string.strip().lower()
        if not ts:
            if empty_is_trustworthy:
                raise BadIdentifierValue(f"This identifier requires a value for {self.request_name}")
            else:
                return [None]*(len(self.elems))
        else:
            return [1.0 if x in ts else 0.0 for x in self.elems]

    def observed_ranges(self):
        return [("0.0 (absent)", "1.0 (present)")] * len(self.elems)


class SporeFeatureParser(ParserBase):

    def __init__(self, db_name, request_name, initial_letter, min_number, max_number):
        self.initial_letter = initial_letter
        self.min_number = min_number
        self.max_number = max_number
        self.multiplier = 1.0
        self.targets = {}
        for x in range(min_number, max_number+1):
            first = self.initial_letter + str(x)
            joined = first + "," + self.initial_letter + str(x+1)
            first_value = (x-min_number)/(max_number-min_number)
            self.targets[first] = first_value
            if x < max_number:
                joined_value = (x+0.5-min_number)/(max_number-min_number)
                self.targets[joined] = joined_value
        self.obs_range = [(f"0.0 ({self.min_number})", f"1.0 ({self.max_number})")]
        super().__init__(db_name, request_name, ["SporeFeature" + self.initial_letter])

    def train(self, _value, _verbose):
        pass

    def transform(self, this_string, empty_is_trustworthy, logger):
        if not this_string.strip():
            if empty_is_trustworthy:
                raise BadIdentifierValue(f"This identifier requires a value for {self.request_name}")
            else:
                return [None]
        target = ",".join([elem.strip() for elem in this_string.strip().upper().replace(";", ",").split(",")])
        if target in self.targets:
            return [self.multiplier * self.targets[target]]
        else:
            if empty_is_trustworthy:
                raise BadIdentifierValue(f"{this_string} is not a valid value for {self.request_name}. Please choose from {self.targets.keys()}")
            else:
                # Ignore for now - we have seen P0,P1,P2 in practice
                return [None]

    def observed_ranges(self):
        return self.obs_range


class CheiloShapeParser(ParserBase):

    def __init__(self, db_name, request_name, include_pyriform=False):
        if include_pyriform:
            super().__init__(db_name, request_name,
                             ["CheiloCylindrical",
                              "CheiloVentiLagen",
                              "CheiloClavateGently",
                              "CheiloClavateStipitate",
                              "CheiloClavateVentricose",
                              "CheiloPyriBalloon"])
            self.impl = CheiloShapes2()
        else:
            super().__init__(db_name, request_name,
                             ["CheiloCylindrical",
                              "CheiloVentiLagen",
                              "CheiloClavateGently",
                              "CheiloClavateStipitate",
                              "CheiloClavateVentricose"])
            self.impl = CheiloShapes()

    def train(self, _value, _verbose):
        pass

    def transform(self, this_string, empty_is_trustworthy, logger):
        return self.impl.parse_to_one_zero_none(this_string, empty_is_trustworthy)

    def observed_ranges(self):
        return [("0.0 (absent)", "1.0 (present)")]*len(self.column_names)


class AssociatesWithParser(ParserBase):

    def __init__(self, request_name):
        super().__init__("associated_ecm_families_objs", request_name, ["Pinaceae", "Salicaceae", "Fagaceae"])

    def train(self, _value, _verbose):
        pass

    def transform(self, values, empty_is_trustworthy, logger):
        objs = values.all()
        is_empty = len(objs) == 0  # no associations recorded
        if is_empty:
            if empty_is_trustworthy:
                return [0.0] * len(self.column_names)
            else:
                return [None] * len(self.column_names)
        else:
            base = [0.0] * len(self.column_names)
            for assoc in objs:
                for ix, col_name in enumerate(self.column_names):
                    if assoc.name.lower() == col_name.lower():
                        logger.write(f"GOT A MATCH ON {assoc.name}")
                        base[ix] = 1.0
            return base

    def observed_ranges(self):
        return [("0.0 (absent)", "1.0 (present)")]*len(self.column_names)


class IgnoreZeroInfNaNParserBase(ParserBase):  # noqa

    def __init__(self, db_name, request_name, figure_display_name):
        self.values = []
        self.is_sorted = False
        self.figure_display_name = figure_display_name
        super().__init__(db_name, request_name, [request_name])

    @staticmethod
    def _is_bad(this_float):
        return (this_float is None) or math.isnan(this_float) or math.isinf(this_float)

    def train_(self, this_float):
        if not self._is_bad(this_float):
            self.values.append(this_float)

    def _ensure_sorted(self):
        if not self.is_sorted:
            self.values.sort()
            self.is_sorted = True

    def min_value(self):
        self._ensure_sorted()
        return self.values[0]

    def max_value(self):
        self._ensure_sorted()
        return self.values[len(self.values) - 1]

    def plot_distribution(self, logger=_BlankLogger()):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        num_points = 1000
        mn = self.min_value()
        mx = self.max_value()
        delta = (mx - mn)/num_points
        xs = []
        ys = []
        for i in range(1+num_points):
            x = mn + i*delta
            y = self._find_implied_value(x, logger)
            xs.append(x)
            ys.append(y)
        ax.plot(xs, ys, marker="+")
        plt.xlabel(self.figure_display_name)
        plt.ylabel("Cumulative probability / Mapped value")
        plt.show()

    def _find_implied_value(self, this_float, logger):
        # A massively simplified alternative would be
        #   return [(this_float - self.min) / (self.max - self.min)]
        # but we don't do this
        #
        # We are in effect constructing a probability distribution
        # from the observed values
        self._ensure_sorted()
        if len(self.values) == 0:
            raise RuntimeError("No values available for interpolation - bad de-pickling?")
        if len(self.values) == 1:
            logger.write(f"{self.request_name} - One values only! Will return 0.5")
            return 0.5
        ll = len(self.values)
        mn = self.values[0]
        mx = self.values[ll-1]
        if (this_float <= mn) or (this_float >= mx):
            # Extrapolate
            return (this_float - mn)/(mx-mn)
        # interpolate
        lindex = bisect.bisect_left(self.values, this_float)
        rindex = bisect.bisect_right(self.values, this_float)
        assert lindex > 0  # The this_float > mn cond ensure this
        assert rindex >= lindex
        if lindex == rindex:
            # lindex : all(val < x for val in a[lo:i])
            # rindex : all(val <= x for val in a[lo:i])
            # Equal left and right insertion indexes
            # => there is only place to insert this

            delta = 1.0/(ll-1)

            low_value = self.values[lindex-1]
            high_value = self.values[lindex]
            local_lambda = (this_float - low_value)/(high_value - low_value)
            return delta*(lindex-1 + local_lambda)
        else:
            # lindex is the least with value=x
            # rindex-1 is the highest with value=x
            delta = 1.0/(ll-1)
            low_range = delta*lindex
            high_range = delta*(rindex-1)
            return (low_range + high_range)/2
            # E.g. if exactly equal to the zero-th element
            # lindex = 0, rindex=1
            # return [delta*0 + delta*0 ]/2 = 0
            # E.g. if exactly equal to first two elements
            # lindex = , rindex=1
            # return [delta/2]

    def transform_(self, this_float, empty_is_trustworthy, logger):
        if not self._is_bad(this_float):
            logger.write(f"{self.request_name}: {this_float} is not bad, will imply a value from {len(self.values)} that have been stored.")
            ret = self._find_implied_value(this_float, logger)
            logger.write(f"{self.request_name}: Returning {ret}")
            return [ret]
        else:
            if empty_is_trustworthy:
                raise BadIdentifierValue(f"This identifier requires a value for '{self.request_name}'")
            else:
                logger.write(f"{self.request_name}: Bad value - returning None")
                return [None]

    def observed_ranges(self):
        self._ensure_sorted()
        ll = len(self.values)
        if ll == 0:
            return [("---", "---")]
        else:
            return [("%.2f" % self.values[0], "%.2f" % self.values[ll-1])]


class IgnoreZeroInfNaNParser(IgnoreZeroInfNaNParserBase):

    def __init__(self, db_name, request_name, figure_display_name):
        super().__init__(db_name, request_name, figure_display_name)
        self.parser = self.compute_core_str_parser()

    def compute_core_str_parser(self):
        from .geography import parse_lng, parse_lat
        if self.request_name.lower() == "latitude":
            return parse_lat
        elif self.request_name.lower() == "longitude":
            return parse_lng
        else:
            return lambda x: float(x)

    def get_core_str_parser(self):
        # Could happen for old, pickled parsers
        p = getattr(self, "parser", None)
        if not p:
            return self.compute_core_str_parser()
        else:
            return self.parser

    def train(self, this_float, _verbose):
        self.train_(this_float)

    def transform(self, incoming, empty_is_trustworthy, logger):
        # When we come from a request we may have a string at this stage
        if isinstance(incoming, str):
            if not incoming:
                if empty_is_trustworthy:
                    incoming = None
                else:
                    raise RuntimeError(f"Blank input not valid for field {self.request_name}")
            else:
                try:
                    incoming = self.get_core_str_parser()(incoming)
                except ValueError:
                    raise RuntimeError(f"Input '{incoming}' not valid for field {self.request_name}") from None
        return self.transform_(incoming, empty_is_trustworthy, logger)


class IgnoreZeroInfNaNParserSToSingle(IgnoreZeroInfNaNParserBase):

    def __init__(self, db_name, request_name, figure_display_name):
        self.preprocessor = SToSingle(request_name)
        super().__init__(db_name, request_name, figure_display_name)

    def train(self, s_string, verbose):
        pp_value = self.preprocessor.parse(s_string)
        if verbose:
            print(f"The preprocessed value is {pp_value}")
        self.train_(pp_value)

    def transform(self, s_string, empty_is_trustworthy, logger):
        pp_value = self.preprocessor.parse(s_string)
        logger.write(f"The preprocessed value is {pp_value}")
        return self.transform_(pp_value, empty_is_trustworthy, logger)


class JustValueParser(ParserBase):

    def __init__(self, db_name, request_name):
        super().__init__(db_name, request_name, column_names=[request_name])

    def train(self, _incoming, _verbose):
        pass

    def transform(self, incoming, empty_is_trustworthy, logger):
        return [incoming]

    def observed_ranges(self):
        return [("Just value", "Just value")]


class IsEqualToStringParser(ParserBase):

    def __init__(self, db_name, request_name, targets):
        self.targets = [t.lower for t in targets]
        super().__init__(db_name, request_name, [request_name + t for t in targets])

    def train(self, _incoming, _verbose):
        pass

    def transform(self, incoming, empty_is_trustworthy, logger):
        inc = incoming.strip().lower()
        if not inc:
            return [0.0 if empty_is_trustworthy else None] * len(self.targets)
        else:
            return [1.0 if inc == target else 0.0 for target in self.targets]

    def observed_ranges(self):
        return [("0.0 (absent)", "1.0 (present)")]*len(self.targets)


class YesNoUnknownParser(ParserBase):

    def __init__(self, db_name, request_name, value_for_unknown):
        super().__init__(db_name, request_name, [request_name])
        self.value_for_unknown = value_for_unknown

    def train(self, _incoming, _verbose):
        pass

    def transform(self, incoming, empty_is_trustworthy, logger):
        inc = incoming.strip().lower()
        if inc == "yes":
            return [1.0]
        elif inc == "no":
            return [0.0]
        elif inc == "variable":
            return [0.5]
        elif inc == "unknown":
            return [self.value_for_unknown]
        else:
            raise RuntimeError(f"{self.request_name} should be yes, no or variable (received {inc})")

    def observed_ranges(self):
        return []


# parsers
core_parsers = [
                # Parser,                                                        everything, include_in_v7, include_in_v8, include_in_9 or 10, include in amyg special, include in 12, include in 13, CG10
                # This is for filtering
                (JustValueParser("continent", "continent"), True, False, False, False, False, False, False, False, False, False, False, False, False, False),

                # These are for regressing
                # (SporeFeatureParser("spore_ornamentation", "SporeOX", "O", 0, 4)      , True, False, False, False, False, False, False, False, False),
                # (SporeFeatureParser("spore_perispore_loosening", "SporePX", "P", 0, 3), True, False, False, False, False, False, False, False, False),
                # (SporeFeatureParser("spore_dextrinoidity", "SporeDX", "D", 0, 4)      , False, True, False, False, False, False, False, False, False),

                (StringContainsParser("spore_ornamentation", "SporeO", ["1", "2", "3", "4"]), True, True, True, True, False, True, True, True, True, True, True, True, False, False),
                (StringContainsParser("spore_perispore_loosening", "SporeP", ["0", "1", "2", "3"]), True, True, True, True, False, True, True, True, True, True, True, True, False, False),
                (StringContainsParser("spore_dextrinoidity", "SporeD", ["0", "1", "2", "3", "4"]), True, True, True, True, False, True, True, True, True, True, True, True, True, True),

                (IgnoreZeroInfNaNParser("spore_length_average_um", "SporeLength", "Spore length (µm)"), True, True, True, True, True, True, True, True, True, True, True, True, False, False),
                (IgnoreZeroInfNaNParser("spore_width_average_um", "SporeWidth", "Spore width (µm)"), True, True, False, False, True, True, True, True, True, True, True, True, True, True),
                (IgnoreZeroInfNaNParser("spore_q_average", "SporeQ", "Spore Q ratio"), True, True, True, True, True, True, True, True, True, True, True, True, False, False),

                (IgnoreZeroInfNaNParser("cheilocystidia_length_average_um", "CheiloLength", "Cheilocystidia length (µm)"), True, True, True, True, False, True, True, True, True, True, True, True, False, False),
                (IgnoreZeroInfNaNParser("cheilocystidia_apex_on_gill_edge_average_um", "CheiloWidth", "Cheilocystidia width (µm)"), True, False, True, True, False, True, True, True, True, False, False, False, False, False),
                (IgnoreZeroInfNaNParser("cheilocystidia_q1_am", "CheiloQ_AM", "Cheilocystidia A/M ratio"), True, False, False, True, False, True, True, True, True, False, False, False, False, False),
                (IgnoreZeroInfNaNParser("cheilocystidia_q2_ab", "CheiloQ_AB", "Cheilocystidia A/B ratio"), True, False, True, True, False, True, True, True, True, False, False, False, False, False),
                (IgnoreZeroInfNaNParser("cheilocystidia_q3_bm", "CheiloQ_BM", "Cheilocystidia B/M ratio"), True, False, True, True, False, True, True, True, True, False, False, False, False, False),
                (IgnoreZeroInfNaNParser("basidia_q_average", "BasidiaQ", "Basidia Q ratio"), True, False, False, False, False, False, False, True, True, False, False, False, False, True),
                (IgnoreZeroInfNaNParserSToSingle("stipe_median_width_mm", "StipeMedianWidth", "Stipe width (mm)"), True, False, False, False, False, False, False, False, False, False, False, False, False, False),
                (IgnoreZeroInfNaNParserSToSingle("number_of_complete_lamellae", "NoOfCompleteLamellae", "No of complete lamellae"), True, True, True, True, True, True, True, True, True, False, False, True, True, True),

                (YesNoUnknownParser("pileus_characters_remains_of_universal_veil", "RemainsOfUniversalVeil", value_for_unknown=None), True, False, False, False, False, False, False, False, False, False, False , False, False, False),
                                                                             # 1    2       3    4     5    6      7    8      9    10     11   12
                (IgnoreZeroInfNaNParser("latitude", "Latitude", "Latitude"), True, True, True, True, True, True, True, True, True, True, True, True, False, False),
                (IgnoreZeroInfNaNParser("longitude", "Longitude", "Longitude"), True, True, True, True, True, True, True, True, True, True, True, True, False, False),
                (IgnoreZeroInfNaNParser("altitude", "Altitude", "Altitude (m)"), True, True, True, True, True, True, True, True, True, True, True, True, False, False),

                # (CheiloShapeParser("cheilocystidia_main_shape", "CheiloShape", include_pyriform=False), False, True, True, True, False, True, False, False),
                (CheiloShapeParser("cheilocystidia_main_shape", "CheiloShape", include_pyriform=True), True, True, True, True, False, False, True, True, True, False, True, False, False, False),

                (YesNoUnknownParser("smell_sacchariolentia", "Smell", value_for_unknown=None), True, False, False, False, False, False, False, False, False, False, False, False, False, False),

                (AssociatesWithParser("AssociatesWith"), True, False, False, False, False, False, False, False, True, False, False, False, False, False)
                ]


def all_parsers():
    return core_parsers


parser_indexes = {"CG4": 2,     # CG names are the names used in the paper == v7
                  "CG5": 3,     # == v8
                  "CG6": 4,     # == v9
                  "CG7": 7,     # == v13
                  "CG8": 8,     # == v14
                  "CG9": 9,     # == v15
                  "CG1": 10,    # no equivalent
                  "CG2": 11,    # has cheilo shape but not lamellae
                  "CG3": 12,    # has lamellae but not shape
                  "CG10": 13,    # Just spore width, dextroidinty, lamellae
                  "CG11": 14,    # CG10 plus basidia q

                  "everything": 1,
                  "v7": 2,
                  "v8": 3,
                  "v9": 4,
                  "v10": 4,
                  "v12": 6,
                  "v13": 7,
                  "v14": 8,   # v13 plus basidia
                  "v15": 9}   # v14 plus assoc


def identifier_all_features_string():
    global core_parsers
    res = set()
    for x in core_parsers:
        res.add(x[0].db_name)
    return ",".join(res)


def _species_is_amyg(species):
    amyg_fullname = "Hebeloma-Amygdalina"
    return species_generalised_section(species) == amyg_fullname


def _make_subclassifier_applies_funcs(parsers_names: Iterable[str]):

    def simple_true(_):
        return True

    global parser_indexes
    res = []
    for parsers_name in parsers_names:
        print(f"Parser name is {parsers_name}")
        if parsers_name.startswith("amygdalina_"):
            res.append(_species_is_amyg)
        elif parsers_name in parser_indexes:
            res.append(simple_true)
        else:
            raise RuntimeError(f"Unable to assign parsers_name {parsers_name} to a subclassifier applies function")
    return res

def _make_filters(include_continent_pre_filter, parsers_names: Iterable[str]):
    if not include_continent_pre_filter:
        return []
    continent_indexes = []
    for ix, _ in enumerate(parsers_names):
        continent_indexes.append(ix)
    # Old way was coupled to parser name
    # for ix, parser_name in enumerate(parsers_names):
    #    if parser_name in ("v10",  "v12", "v13", "v14", "v15", amyg):
    #        continent_indexes.append(ix)
    if continent_indexes:
        return [("continent", continent_indexes)]
    else:
        return []


def _parser_name_to_cg_index(parser_name):
    global parser_indexes
    if parser_name.startswith("amygdalina_"):
        pn = parser_name[11:]
    else:
        pn = parser_name
    return parser_indexes[pn]


def _make_parsers(parsers_names: Iterable[str]):
    global core_parsers
    result = []
    indexes = list([_parser_name_to_cg_index(parser_name) for parser_name in parsers_names])
    for parser_config in core_parsers:
        these_indexes = []
        for ix, _ in enumerate(parsers_names):
            index_into_core_data = indexes[ix]
            if index_into_core_data >= len(parser_config):
                raise RuntimeError(f"{parser_config[0].request_name} doesn't accommodate index {index_into_core_data}")
            if len(parser_config) != 15:
                raise RuntimeError(f"{parser_config[0].request_name} sanity check failed - expect all parser defs to be length 15 right now")

            if parser_config[index_into_core_data]:
                these_indexes.append(ix)
        if these_indexes:
            # Ok at least one of the subclassifiers is going to use
            # this core parser
            result.append((parser_config[0], these_indexes))
    return result



class HebelomaTrainer:

    def __init__(self,
                 parsers_names: Sequence[str],
                 training_set_proportion,
                 missing_allowance,
                 require_full_testing_data,
                 include_continent_pre_filter: bool):

        # Record which collection IDs were used for training
        # just for future display
        self.training_set_ids = []
        self.testing_set_ids = []
        self.training_set_proportion = training_set_proportion
        self.missing_allowance = missing_allowance
        self.require_full_testing_data = require_full_testing_data
        self.parsers_names = parsers_names

        # This is the fun bit we have a list of parsers

        # A list of pairs of 2-tuples
        # The first element is the string name of the filter column
        # The second element is a list of classifier indexes to which the filter will be applied
        self.filter_columns_master = _make_filters(include_continent_pre_filter, parsers_names)

        # A similar list of pairs of 2-tuples
        # The difference is that the first item is a full-on parser, not just a string
        self.feature_parsers_master = _make_parsers(parsers_names)

        self.num_classifiers = len(parsers_names)  # noqa
        self.training_set_row_indexes = []
        self.testing_set_row_indexes = []
        for _ in range(self.num_classifiers):
            self.training_set_row_indexes.append([])
            self.testing_set_row_indexes.append([])

        self.subclassifier_applies_funcs = _make_subclassifier_applies_funcs(parsers_names)

    def mark_in_training_set(self, specimen_number, applicable_subclassifiers: Iterable[bool]):
        training_data_ix = len(self.training_set_ids)
        self.training_set_ids.append(specimen_number)
        for ix, is_applicable in enumerate(applicable_subclassifiers):
            if is_applicable:
                self.training_set_row_indexes[ix].append(training_data_ix)

    def mark_in_testing_set(self, specimen_number, applicable_subclassifiers: Iterable[bool]):
        testing_data_ix = len(self.testing_set_ids)
        self.testing_set_ids.append(specimen_number)
        for ix, is_applicable in enumerate(applicable_subclassifiers):
            if is_applicable:
                self.testing_set_row_indexes[ix].append(testing_data_ix)

    def training_set_row_indexes_for(self, classifier_ix):
        return self.training_set_row_indexes[classifier_ix]

    def testing_set_row_indexes_for(self, classifier_ix):
        return self.testing_set_row_indexes[classifier_ix]

    def filter_columns(self, classifier_ix):
        return [fc[0] for fc in self.filter_columns_master if classifier_ix in fc[1]]

    def all_regression_columns(self):
        res = []
        for parser, _ in self.feature_parsers_master:
            res += parser.column_names
        return res

    def feature_parsers(self, classifier_ix):
        return [parser for parser, indexes in self.feature_parsers_master if classifier_ix in indexes]

    def regression_columns(self, classifier_ix):
        res = []
        for p in self.feature_parsers(classifier_ix):
            res += p.column_names
        return res

    @staticmethod
    def target_columns():
        return ["Species", "Section"]

    def train_from_coll(self, coll, verbose):
        # When training from collection
        # We get all the data for all filters and all parsers
        # at once (compare below for transforming from request, which is per classifier)
        for parser, _ in self.feature_parsers_master:
            parser.train_from_coll(coll, verbose)

    def transform_from_coll(self, coll, classifier_number):
        # E.g. called from collections.html
        filter_res = [getattr(coll, column_name) for column_name in self.filter_columns(classifier_number)]
        regress_res = []
        for p in self.feature_parsers(classifier_number):
            regress_res += p.transform_from_coll(coll, verbose=False)
        return filter_res, regress_res

    def data_from_coll(self, coll, classifier_number):
        keys = self.filter_columns(classifier_number)
        values = [getattr(coll, column_name) for column_name in self.filter_columns(classifier_number)]
        for p in self.feature_parsers(classifier_number):
            this_key, this_value = p.data_from_coll(coll)
            keys.append(this_key)
            values.append(this_value)
        return keys, values

    def transform_from_request(self, input_getter, classifier_number, logger):
        filter_res = []
        logger.write(f"Transforming for classifier {classifier_number}")
        for field in self.filter_columns(classifier_number):
            value = input_getter(field)
            filter_res.append(value)
            logger.write(f"Filter field {field} has value {value}")
        regress_res = []
        for p in self.feature_parsers(classifier_number):
            value = p.transform_from_req(input_getter, logger)
            regress_res += value
        return filter_res, regress_res

    def observed_ranges(self):
        res = []
        for parser, _ in self.feature_parsers_master:
            res += parser.observed_ranges()
        return res

    def train(self, collections_df):
        filter_columns = [column_name for column_name, _ in self.filter_columns_master]
        regression_columns = self.all_regression_columns()
        target_columns = self.target_columns()

        training_data = []
        testing_data = []

        num_filter_columns = len(filter_columns)
        num_regression_columns = len(regression_columns)
        num_target_columns = len(target_columns)
        minimal_columns = num_filter_columns + num_regression_columns + num_target_columns - self.missing_allowance
        to_be_populated = num_regression_columns - self.missing_allowance

        print("Doing transforming pass.")
        print(f"There are a total of {len(collections_df)} collections to select from.")
        print(f"There are a total of {num_filter_columns} filter columns.")
        print(f"There are a total of {num_regression_columns} regression columns.")
        print(f"Require {to_be_populated} regression columns to be populated")

        # We want to balance EACH species to have (as close as possible)
        # the required proportion in the training set
        # so we can't do a simple random assignment

        species_to_colls = defaultdict(list)
        colls_to_data = {}

        # TODO: This is a bit flaky
        # It assume that the most "finely-grained" group (i.e species)
        # is the first in the list of training columns
        # This only holds by convention
        fine_grouping_index = -num_target_columns
        for ix, row in collections_df.iterrows():
            specimen_number = row["SpecimenID"]

            # Extra-ordinarily inefficient!
            this_data = []
            for filter_col in filter_columns:
                this_data.append(row[filter_col])
            for regression_col in regression_columns:
                this_data.append(row[regression_col])
            for target_col in target_columns:
                this_data.append(row[target_col])
            # print("========")
            # print(this_data)
            # print("========")

            provided_count = sum(1 if (isinstance(td, str) or (0.0 <= td <= 1.0)) else 0 for td in this_data)
            if provided_count >= minimal_columns:
                # print(f"Transformed collection {ix}. It has {provided_count} (>= {minimal_columns}) relevant fields populated so will be included")
                species = this_data[fine_grouping_index]
                species_to_colls[species].append(specimen_number)
                colls_to_data[specimen_number] = this_data
            # else:
            #    print(f"Transformed collection {ix}. It has {provided_count} (< {minimal_columns}) relevant fields populated so will NOT be included")

        # Now split the data into training and test sets

        random.seed(25)
        testing_rejected_count = 0
        for species, colls in species_to_colls.items():
            total_samples = len(colls)
            number_to_train = round(self.training_set_proportion*total_samples)
            training_colls = random.sample(colls, number_to_train)

            applicable_subclassifiers = [sa(species) for sa in self.subclassifier_applies_funcs]

            # For the elements in the training set
            # if we have allowed any elements of the data to not be supplied
            # replace the None by the average of the values
            # of the other elements in the training set for this species

            sums = [[0, 0]] * num_regression_columns
            for coll in training_colls:
                this_data = colls_to_data[coll][num_filter_columns:num_filter_columns+num_regression_columns]
                assert(len(this_data) == num_regression_columns)
                for i in range(num_regression_columns):
                    td = this_data[i]
                    if 0.0 <= td <= 1.0:
                        sums[i][0] += this_data[i]
                        sums[i][1] += 1
            avgs = [0.5 if sum_i[1] == 0 else sum_i[0]/sum_i[1] for sum_i in sums]

            training_count = 0
            testing_count = 0

            for coll in colls:
                this_data = colls_to_data[coll]
                if coll in training_colls:
                    for i in range(num_regression_columns):
                        td = this_data[num_filter_columns+i]
                        if not (0.0 <= td <= 1.0):
                            this_data[num_filter_columns+i] = avgs[i]

                    training_data.append(this_data)
                    self.mark_in_training_set(coll, applicable_subclassifiers)
                    training_count += 1
                else:
                    # Whether to include incomplete data in the testing set
                    # is now toggleable on the input
                    include_it = True
                    if self.require_full_testing_data:
                        reject = False
                        for i in range(num_regression_columns):
                            td = this_data[num_filter_columns + i]
                            if not (0.0 <= td <= 1.0):
                                reject = True
                                # print(f"Rejecting {coll}. i={i}, td={td}")
                        include_it = not reject

                    if include_it:
                        testing_data.append(this_data)
                        self.mark_in_testing_set(coll, applicable_subclassifiers)
                        testing_count += 1
                    else:
                        testing_rejected_count += 1

        all_columns = filter_columns + regression_columns + target_columns
        training_df = pd.DataFrame(training_data, columns=all_columns)
        testing_df = pd.DataFrame(testing_data, columns=all_columns)
        print(f"Finished gathering training/testing sets")
        print(f"There are {len(training_data)} training collections.")
        print(f"There are {len(testing_data)} testing collections.")
        print(f"{testing_rejected_count} were rejected from testing because of incomplete data")
        return training_df, testing_df


class NoFilterFilterFunc:

    @staticmethod
    def should_filter(_a, _b):
        return False


class ValueSeenFilterFunc:
    """
    This is a 'filter func' in the sense of self.filter_func
    in ClassifierEx

    An example use would be 'continent'
    During the training phase we build up a list of
    continents where a species has been seen

    Then, during classification, we know where the to-be-classified collection
    is from. If a species has never been found there, then we simply
    filter out that possibility AFTER the primary classification has been done.
    """
    def __init__(self):
        self.seen_values = defaultdict(set)

    # During the training phase we build up the seen values
    # Discussion point - training set or all collections??
    def add_seen(self, class_, seen_value):
        self.seen_values[seen_value].add(class_)

    def should_filter(self, filter_values, target_class):
        if len(filter_values) != 1:
            raise RuntimeError("For ValueSeenFilterFunc expected one value")
        value = filter_values[0]
        if value not in self.seen_values:
            # We have never seen these value before.
            # (E.g. a new continent).. don't filter, instead
            # accept that any species is a possibility
            return False
        # I.e. filter it out if we never saw that value
        return target_class not in self.seen_values[value]


# THIS NEEDS A SPECIES <-> SPECIES GENERALISED SECTION <-> CONTINENTS map
def _make_continent_filter_func():
    from .parse import parse_formatted_string
    res = ValueSeenFilterFunc()
    for sp in Species.objects.all():  # noqa  # This is ok, the higher level filtering on species in the identifier will prevent any unallowed species from kicking in.
        seen_continents = parse_formatted_string(sp.continent).keys()
        section = species_generalised_section(sp)
        for cont in seen_continents:
            # We only trust our Europe and Northern America data
            # For all the others we allow all possible species
            if cont.lower() in ("europe", "northern america"):
                res.add_seen(sp.name, cont)
                res.add_seen(section, cont)
    return res

class SubClassifier:

    def __init__(self, parser_name, raw_classifier, filter_columns, redo_threshold_for_secondary_classifiers):
        self.parser_name = parser_name
        self.raw_classifier = raw_classifier
        self._redo_threshold_value = redo_threshold_for_secondary_classifiers

        if len(filter_columns) == 0:
            self.filter_func = NoFilterFilterFunc()
        elif len(filter_columns) == 1:
            if filter_columns[0] == "continent":
                self.filter_func = _make_continent_filter_func()
            else:
                raise RuntimeError(f"Don't handle how to handle a filter column '{filter_columns[0]}'")
        else:
            raise RuntimeError(f"Expected exactly 0 or 1 filter funcs, not {len(filter_columns)}")

    def _redo_threshold(self):
        # Written this way to cope with old deserialized classifiers that had the value hard-coded
        # Rather than an attribute on the class
        if hasattr(self, "_redo_threshold_value"):
            return self._redo_threshold_value
        else:
            return 0.9

    def _basic_classify_one(self, filter_values, regress_values):
        # Missing points are None or NaN if they have come from a dataframe
        def is_bad(x):
            if x is None:
                return True
            if not isinstance(x, float):
                raise RuntimeError(f"Regression value '{x}' (type ({type(x)}) is invalid. All regression values were {regress_values}")
            if math.isnan(x):
                return True
            return False

        bad_count = sum([1 if is_bad(x) else 0 for x in regress_values])
        class_names = self.raw_classifier.class_names()

        if bad_count == 0:
            # All values supplied
            # We don't have to integrate across the missing values
            totals = self.raw_classifier.predict(regress_values)
        elif bad_count == 1:
            replacements = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            totals = [0.0] * len(class_names)
            for replacement in replacements:
                this_predict = [x if not is_bad(x) else replacement for x in regress_values]
                this_predict = self.raw_classifier.predict(this_predict)
                for i in range(len(totals)):
                    totals[i] += this_predict[i]
            for i in range(len(totals)):
                totals[i] /= len(replacements)
        else:
            raise RuntimeError(f"Cannot make a prediction - {bad_count} missing values is too many")

        chances = zip(class_names, totals)
        return self._apply_post_filter(filter_values, chances)

    def _apply_post_filter(self, filter_values, chances):
        remaining_prob = 0.0
        result = []
        for class_name, prob in chances:
            # Filter func returns True if SHOULD filter
            # I.e. we know just from the filter value
            # That we can't possibly be this class
            if self.filter_func.should_filter(filter_values, class_name):
                result.append([class_name, 0.0])
            else:
                result.append([class_name, prob])
                remaining_prob += prob
        # There's a possible pathological case
        # Where the classifier has assigned zero probability
        # to all the non-filtered. This would be a bad sign,
        # but is possible..
        # Otherwise rescale up-to 1.0
        # (A more sophisticated approach would not reduce
        #  all filtered out fields prob by 100% but by some smaller
        #  configurable fraction)
        if remaining_prob > 0:
            for r in result:
                r[1] = r[1] / remaining_prob
        return result

    def classify_impl(self, filter_values, regress_values, chances_so_far, verbose=False):
        if not chances_so_far:
            # We are the first classifier in the chain...
            # The chances are just what we say they are
            return [self._basic_classify_one(filter_values, regress_values)]
        else:
            # We are later in the chain and we have some results from an earlier classifier
            if not self.parser_name.startswith("amygdalina"):
                raise RuntimeError(f"Only amygdalina is currently supported as a secondary classifier. Received {self.parser_name}")
            # chances_so_far is a list of an unsorted iterable of two-tuples
            # (species_name, prob)
            # We take the probabilities so far from the last element of the list
            total_amyg_prob = sum([p for name, p in chances_so_far[-1] if _species_is_amyg(name)])
            total_prob = sum([p for name, p in chances_so_far[-1]])

            if verbose:
                print(f"Doing amygdalina second pass - total_prob {total_prob} - total_amgy_prob {total_amyg_prob}")
            redo_threshold = self._redo_threshold()
            if total_amyg_prob > redo_threshold:
                # exceeds_threshold_count += 1
                # print(f"Exceeds re-do threshold")
                sub_probs = self._basic_classify_one(filter_values, regress_values)
                new_results_pure = []
                # new_results_blended = []
                for name, old_p in chances_so_far[-1]:
                    if _species_is_amyg(name):
                        # We try two classifiers
                        # In the first case we entirely replace the
                        # old prob from the first classifier with the new
                        # prob from the second classifier
                        # In the second case we try the average of the two
                        # There is no rationale for this - we are just trying it
                        new_p = 0.0
                        for amyg_name, this_amyg_prob in sub_probs:
                            if name == amyg_name:
                                new_p = this_amyg_prob * total_amyg_prob
                                break
                        new_results_pure.append((name, new_p))                      # case 1
                        # new_results_blended.append((name, (new_p+old_p)/2))       # case 2
                    else:
                        new_p = old_p
                        new_results_pure.append((name, new_p))
                    if verbose:
                        print(f"For {name}, old_p = {old_p}, new_p = {new_p}")

                chances_so_far.append(new_results_pure)
            else:
                if verbose:
                    print(f"Does not exceed re-do threshold")
                # amyg chance doesn't exceed threshold
                # so just return old results
                chances_so_far.append(chances_so_far[0])

            if verbose:
                import pprint
                pprint.pprint(chances_so_far)
            return chances_so_far


class ClassifierEx:

    """
    This object is the union of a real classifier from PyTorch
    and the column names of training data
    and a transformer to get from
    real data to scaled data
    """
    def __init__(self, subclassifiers, trainer_impl: HebelomaTrainer, context: str, target_type: str):
        self.subclassifiers = subclassifiers
        self.trainer_impl = trainer_impl
        self.context = context
        self.target_type = target_type
        self.all_classification_results = defaultdict(list)

    def field_used(self, field_name):
        db_names = [parser.db_name for parser, _ in self.trainer_impl.feature_parsers_master]
        return field_name in db_names

    def coll_id_to_set_type(self, coll_id):
        # ONE **MUST** NOT USE the collection ID
        # (if supplied) in the classifier.. that would be cheating!!
        coll_id_class = COLL_ID_NOT_PRESENT
        if coll_id:
            present = coll_id in self.trainer_impl.training_set_ids
            coll_id_class = COLL_ID_IN_TRAINING_SET if present else COLL_ID_NOT_IN_TRAINING_SET
        return coll_id, coll_id_class

    def _classify_from_request_or_coll(self, transform_method):
        chances_so_far = []
        for ix, subc in enumerate(self.subclassifiers):
            filter_values, regress_values = transform_method(ix)
            chances_so_far = subc.classify_impl(filter_values, regress_values, chances_so_far)
        # Return the last classification
        return chances_so_far[-1]

    def classify_from_request(self, input_getter, logger, confidence_alpha):

        def transformer(ix):
            return self.trainer_impl.transform_from_request(input_getter, ix, logger)

        untuned_result = self._classify_from_request_or_coll(transformer)
        logger.write("Untuned result")
        for n, p in untuned_result:
            logger.write(f"{n}: {p}")
        # Recall that we have probabilities derived from scores as if
        # P(i) = exp(S(i)) / Sum_j exp(S(j))
        # If we, post-hoc, think that these probabilities are too confident
        # then we can de-tune them by applying a confidence parameter alpha
        # P'(i) = exp(alpha*S(i)) / Sum_j exp(alpha*S(j))
        # where alpha = 0: No confidence at all: Give all species equal probability
        #       alpha < 1: Less confident then original prediction
        #       alpha = 1: Same confidence
        #       alpha > 1: More confident than original prediction (not recommended!)
        if confidence_alpha <= 0.0:
            raise RuntimeError(f"Bad confidence alpha: {confidence_alpha}")
        elif confidence_alpha == 1.0:
            logger.write("Confidence alpha is 1 - returning untuned result")
            return untuned_result
        else:
            from . import invert_softmax
            probs = [p for _, p in untuned_result]
            new_probs = invert_softmax.invert_softmax(probs, confidence_alpha)
            new_result = [(name, new_probs[i]) for i, (name, old_prob) in enumerate(untuned_result)]
            logger.write("Tuned result")
            for n, p in new_result:
                logger.write(f"{n}: {p}")
            return new_result

    def classify_from_coll(self, coll):

        def transformer(ix):
            return self.trainer_impl.transform_from_coll(coll, ix)

        return self._classify_from_request_or_coll(transformer)

    def data_from_coll(self, coll):
        # A palaver to return things (more or less) in "feature order"
        key_res = []
        value_res = []
        for ix, _ in enumerate(self.subclassifiers):
            this_keys, this_values = self.trainer_impl.data_from_coll(coll, ix)
            for this_key, this_value in zip(this_keys, this_values):
                if this_key not in key_res:
                    key_res.append(this_key)
                    value_res.append(this_value)
        return zip(key_res, value_res)

    @staticmethod
    def _by_id_to_by_class_species(by_id):
        # The value against each species is going to be a tuple:
        # 1) generalised_section
        # 2) total
        # 3) first guess
        # 4) second guess
        # 5) third guess
        # 6) fourth guess
        # 7) fifth guess
        # 8) lower guess (total)
        # 9) section correct count
        # 10) section top3 count
        # 11) others dict - if not top 1, what else was guessed

        others_ix = 10
        res = dict()

        def _ensure_species(species_):
            if species_ in res:
                return

            def zero():
                return 0

            #                                                      t  1  2  3  4  5  lower, s1, s3
            initial_value = [species_generalised_section(species_), 0, 0, 0, 0, 0, 0, 0, 0, 0, defaultdict(zero)]
            res[species_] = initial_value

        for ix, species, guesses, guess_prob in by_id.values():
            _ensure_species(species)
            res[species][1] += 1  # total always increments
            if ix != 0:
                # We didn't get the guess exactly right
                # so increment the "guessed as" value
                res[species][others_ix][guesses[0][0]] += 1
            if ix == 0:  # First
                res[species][2] += 1
            elif ix == 1:  # Second
                res[species][3] += 1
            elif ix == 2:  # Third
                res[species][4] += 1
            elif ix == 3:
                res[species][5] += 1
            elif ix == 4:
                res[species][6] += 1
            else:
                res[species][7] += 1

            # Now deal with section correctness
            # We record whether
            #   -   our top guess was the correct section
            #   -   any of our top three guesses were in the correct section
            ss_key = res[species][0]
            is_top1 = ix == 0 or ss_key == species_generalised_section(guesses[0][0])
            is_top3 = is_top1 or ss_key == species_generalised_section(guesses[1][0]) or ss_key == species_generalised_section(guesses[2][0])
            if is_top1:
                res[species][8] += 1
            if is_top3:
                res[species][9] += 1

        return res

    @staticmethod
    def _by_id_to_by_class_section(by_id):
        # The value against each species is going to be a tuple:
        # 1) total
        # 2) first guess
        # 3) second guess
        # 4) third guess
        # 5) fourth guess
        # 6) fifth guess
        # 7) lower guess (total)
        # 8) others dict - if not top 1, what else was guessed

        others_ix = 7
        res = dict()

        def _ensure_section(species):
            if species in res:
                return

            def zero():
                return 0

            #                t  1  2  3  4  5  lower
            initial_value = [0, 0, 0, 0, 0, 0, 0, defaultdict(zero)]
            res[species] = initial_value

        for ix, section, guesses, _ in by_id.values():
            _ensure_section(section)
            res[section][0] += 1  # total always increments
            if ix != 0:
                # We didn't get the guess exactly right
                # so increment the "guessed as" value
                res[section][others_ix][guesses[0][0]] += 1
            if ix == 0:  # First
                res[section][1] += 1
            elif ix == 1:  # Second
                res[section][2] += 1
            elif ix == 2:  # Third
                res[section][3] += 1
            elif ix == 3:
                res[section][4] += 1
            elif ix == 4:
                res[section][5] += 1
            else:
                res[section][6] += 1

        return res

    def _by_id_to_by_class(self, by_id, class_is_species):
        return self._by_id_to_by_class_species(by_id) if class_is_species else self._by_id_to_by_class_section(by_id)

    def _probabilities_to_rankings_one(self, predictions_prob, real_dfs, ids, class_is_species, classifier_ix, verbose_ids):
        """

        Args:
            predictions_prob:
            real_dfs:
            ids:
            class_is_species:
            classifier_ix:

        Returns:
            A 3-tuple:
                by_id: A dictionary of specimen ID to 4-tuple: guess position, real species name, all ids to probs, guess prob
                by_class:
                positions_array: The number of guesses at each position as an array [150, 20, 5, 0, 0, ..., 0] we hope!
        """
        class_names = self.subclassifiers[0].raw_classifier.class_names()
        positions_array = [0]*(len(class_names)+1)
        not_found_ix = len(class_names)
        by_id = dict()

        print(f"Checking {len(predictions_prob)} results")
        for real, names_and_probs_list, idd in zip(real_dfs, predictions_prob, ids):
            # Sort the classes in decreasing order of probability
            # If the real class name was the (cnt+1)-th highest probability
            # store that position
            names_and_probs = names_and_probs_list[classifier_ix]
            res = sorted(names_and_probs, key=lambda x: x[1], reverse=True)

            if idd in verbose_ids:
                import pprint
                pprint.pprint(f"For id: {idd}")
                pprint.pprint(res)

            guess_ix = not_found_ix
            guess_prob = 0.0
            for ix, (name, prob) in enumerate(res):
                if real == name:
                    # We got a match at this ix
                    guess_ix = ix
                    guess_prob = prob
                    break

            positions_array[guess_ix] = positions_array[guess_ix] + 1
            by_id[idd] = guess_ix, real, res, guess_prob

        by_class = self._by_id_to_by_class(by_id, class_is_species)
        return by_id, by_class, positions_array

    def _probabilities_to_rankings(self, predictions_prob, real_dfs, ids, class_is_species, verbose_ids):
        number_of_predictions = len(predictions_prob[0])
        return [self._probabilities_to_rankings_one(predictions_prob, real_dfs, ids, class_is_species, ix, verbose_ids) for ix in range(number_of_predictions)]

    def classify_from_df_one_target(self, ids, df, class_is_species):

        # verbose_ids = 14304,
        verbose_ids = []

        number_to_classify = df.shape[0]
        filter_dfs = []
        predict_dfs = []
        for ix, _ in enumerate(self.subclassifiers):
            filter_dfs.append(df[self.trainer_impl.filter_columns(ix)])
            predict_dfs.append(df[self.trainer_impl.regression_columns(ix)])

        # This looks bizarre - why predict one-by-one
        # when the sklearn API
        # works best with multiple at once?
        # -> because we need to copy with Nones...
        predictions_prob = []
        for i in range(number_to_classify):

            verbose = int(ids[i]) in verbose_ids

            this_prediction = []
            for j, subc in enumerate(self.subclassifiers):
                this_prediction = subc.classify_impl(filter_dfs[j].iloc[i], predict_dfs[j].iloc[i], this_prediction, verbose=verbose)
            predictions_prob.append(this_prediction)

        real_dfs = df[self.target_type]  # Select one column
        return self._probabilities_to_rankings(predictions_prob, real_dfs, ids, class_is_species, verbose_ids)

    def classify_from_df(self, df, ids, storage_key, class_is_species):
        for by_id, by_class, positions_array in self.classify_from_df_one_target(ids, df, class_is_species):
            res = dict()
            res["positions_array"] = positions_array
            res["by_id"] = by_id
            res["by_class"] = by_class
            # This bit used for future display on the website
            self.all_classification_results[storage_key].append(res)
        return self.all_classification_results[storage_key]

    def alphas(self):
        return [subc.classifier_impl.alpha for subc in self.subclassifiers]

    def num_layers(self):
        return [subc.classifier_impl.hidden_layer_sizes[0] for subc in self.subclassifiers]

    def training_set_size(self):
        return len(self.trainer_impl.training_set_ids)

    def testing_set_size(self):
        return len(self.trainer_impl.testing_set_ids)
