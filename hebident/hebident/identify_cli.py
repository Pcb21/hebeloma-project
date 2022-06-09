import argparse
import sys
from .util import BlankLogger, sort_and_thin_probs
from .loader import load_thinned_identifier


class InputValueRetrieverFromStandardIn:

    def __call__(self, field):
        value = input(f"Please provide a value for {field} : ")
        return value


def main():
    # There should be example identifiers in hebident/hebident/examples/
    # So you should be able to cd to the 'hebident' root (parent of the directory containing this file)
    # and execute
    # python -m hebident.identify_cli "hebident\examples\cg7_example_species.dill"
    parser = argparse.ArgumentParser(description='Hebeloma species identification command line tool.')
    parser.add_argument('file', type=str, nargs='?', help='Path to a .dill file containing a saved-down identifier')
    parser.add_argument('--type', type=str, default="species", help='The type of the identifier (either species or section, defaults to species')
    args = parser.parse_args()
    input_getter = InputValueRetrieverFromStandardIn()
    filename = args.file
    is_species = args.type.lower() == 'species'
    classifier = load_thinned_identifier(filename, is_species)
    chances = classifier.classify_from_input_getter(input_getter=input_getter,
                                                    logger=BlankLogger,
                                                    confidence_alpha=0.5,
                                                    empty_is_trustworthy=True,
                                                    use_dbname_not_request_name=False)

    chances = sort_and_thin_probs(chances, number_to_show=5)
    name_width = max(len(c[0]) for c in chances)
    prob_width = max(len(c[1]) for c in chances)

    def padded_name(x):
        spaces = " "*(1+(name_width - len(x)))
        return f" {x}{spaces}"

    def padded_prob(x):
        spaces = " "*(1+(prob_width - len(x)))
        return f" {x}{spaces}"

    total_width = 7 + name_width + prob_width
    class_word = "Species" if is_species else "Section"
    prob_word = "Prob."
    print('')
    print('Identification complete; suggested species are')
    print('')
    print('-'*total_width)
    print(f"|{padded_name(class_word)}|{padded_prob(prob_word)}|")
    print('-'*total_width)
    for name, prob in chances:
        print(f"|{padded_name(name)}|{padded_prob(prob)}|")
    print('-'*total_width)


if __name__ == '__main__':
    sys.exit(main())
