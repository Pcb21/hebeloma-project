import sys
import setuptools


def main():
    dependencies = "dill", "torch", "pandas", "numpy", "reverse_geocoder", "scipy"
    setuptools.setup(name='hebident',
                     packages=setuptools.find_packages(),
                     version='0.0.1',
                     install_requires=dependencies,
                     author='Pete Bartlett',
                     author_email='pete@hebeloma.org',
                     description="A package for Hebeloma species identification",
                     long_description=open('README.md').read(),
                     long_description_content_type='text/markdown',
                     url="http://hebeloma.org",
                     classifiers=["Programming Language :: Python",
                                  "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
                                  "Operating System :: OS Independent",
                                  "Natural Language :: English"])
    return 0


if __name__ == '__main__':
    sys.exit(main())
