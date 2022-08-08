from setuptools import setup, find_packages

VERSION = "0.0.1" 
DESCRIPTION = "Monotonic Binning for Credit Rating Models"
LONG_DESCRIPTION = """Performs monotonic binning of numeric risk factor in credit rating models (PD, LGD, EAD)
	development. All functions handle both binary and continuous target variable.
	Functions that use isotonic regression in the first stage of binning process have an additional 
	feature for correction of minimum percentage of observations and minimum target rate per bin.	
	Additionally, monotonic trend can be identified based on raw data or, if known in advance,
	forced by functions' argument. Missing values and other possible special values are treated
	separately from so-called complete cases.
	This is replica of R 'monobin' package."""

setup(
    name = "monobinpy",
    version = VERSION,
    description = 'Setting up a python package',
    author = "Andrija Djurovic",
    author_email = "djandrija@gmail.com",
    url = "https://github.com/andrija-djurovic/monobinpy",
    licence = "GPL",
    packages = ["monobinpy"],
    package_dir = {"monobinpy": "src/monobinpy"},
    install_requires = [
        "numpy",
        "pandas >= 1.4.3",
	"scipy",
	"scikit-learn",
	"statsmodels"
    ]
)
