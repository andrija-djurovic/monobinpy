## monobinpy
The goal of the monobinpy package is to perform monotonic binning of numeric risk factor in credit 
rating models (PD, LGD, EAD) development. All functions handle both binary and 
continuous target variable. Missing values and other possible special values are treated 
separately from so-called complete cases.
This is replica of monobin R package.

## Installation
To install pypi.org version run the following code:
```cmd
pip install monobinpy
```
and to install development (github) version run:
```cmd
pip install git+https://github.com/andrija-djurovic/monobinpy.git#egg=monobinpy
```

## Example

This is a basic example which shows you how to solve a problem of monotonic binning of numeric risk factors:

```python
import monobinpy as mb
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/andrija-djurovic/monobinpy/main/gcd.csv"
gcd = pd.read_csv(filepath_or_buffer = url)
gcd.head()

res = mb.sts_bin(x = gcd.age.copy(), y = gcd.qual.copy())
res[0]
res[1].value_counts().sort_index()

```
Besides above example, additional five binning  algorithms are available. For details and additional description please check:
```python
help(mb) 
```
