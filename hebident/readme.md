# Welcome to hebident

## About *Hebeloma*

Hebeloma is a genus of fungi that form part of the "wood wide web": They are ectomycorrhizal mushrooms, so they are symbiotic with various trees and shrubs, attaching themselves to the roots of these plants and exchanging nutrients. They form part of the underground network which exists throughout most of the world’s woodlands and forests. Indeed, in many of the world’s temperate forests, Hebeloma are one of the most important and most common groups of fungi.

You can read more about *Hebeloma* at [hebeloma.org](https://hebeloma.org)

## About hebident

hebident is the open source software that powers the species identifier part of the [hebeloma.org](https://hebeloma.org).
The identifier tries to use morphological and geographic data (only) about a *Hebeloma* specimen in order to make a determination as to what particular species it might.

The full methodology is given in the paper *Species determination using AI machine-learning algorithms – Hebeloma as a case study* currently under submission to IMAFungus.

If you just wish to use the identifier tool without looking at the code, the most convenient place to do so is
via the website at [https://hebeloma.org/identifier](https://hebeloma.org/identifier). 
If the website prompts you for a login, please do get in touch at [pete@hebeloma.org](mailto:pete@hebeloma.org).

## Installing hebident

hebident is a Python package which depends on 
1. Python 3.7 or a later version.
1. Any recent version of the pytorch, pandas, numpy, scipy, dill and reverse-geocoder packages. 

To install hebident 
1. clone the main branch of the hebeloma-project repository
2. This should leave with a path like c:\path\to\hebeloma-project\hebident or /path/to/hebeloma-project/hebident
3. In a command line like environment cd to that path
4. Assuming Python is on your PATH with pip installed, execute "python -m pip install -e hebident"
5. This will do a local or "editable" install of hebident into the invoked copy of Python (i.e. 'import hebident' will work)

hebident consists only of simple Python scripts so should work anywhere where its dependencies work, but has only been tested on Windows and Linux

Although hebident uses the machine learning package PyTorch its computational requirements are modest - it should work on any recent PC/laptop.

## Using an identifier from the command line

You can run the identifier from a command line with the following command 

```
python -m hebident.identify_cli <path to saved identifier - a .dill file> --type={'species' or 'section'}
```

This command will prompt you to type in all the feature values (such as average spore width) that are required
by the identifier. Once complete, it will emit the top 5 most likely Hebeloma species for the given values.

An example identifier is given in hebident/examples/cg7_example_species.dill

## Creating your own identifier 

You can create your own identifiers from a command line with the following command 
```
python -m hebident.create_cli <path to collection data - a .csv file>
```
TODO: Document the format of the CSV file here

Some example data is given in hebident/examples/collection_examples.csv

TODO: ACTUALLY CREATE THE EXAMPLE DATA!!!

## Licence

This software is licenced for use under the GNU General Public License v3.

## Contact

For any queries relating to the software, please contact pete@hebeloma.org, thank you.









