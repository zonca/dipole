from setuptools import setup, find_packages
setup(
    name = "dipole",
    version = "0.1",
    packages = ['dipole'], 

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires = ['docutils>=0.3'],

    package_data = {
        # If any package contains *.txt or *.rst files, include them:
        '': ['*.txt', '*.rst'],
    },

    # metadata for upload to PyPI
    author = "Andrea Zonca",
    author_email = "code@andreazonca.com",
    description = "Solar/orbital dipole estimation for Planck",
    license = "BSD",
    keywords = "Planck science data",

    # could also include long_description, download_url, classifiers, etc.
)
