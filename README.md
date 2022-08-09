## monobinpy
The goal of the monobinpy package is to perform monotonic binning of numeric risk factor in credit 
rating models (PD, LGD, EAD) development. All functions handle both binary and 
continuous target variable. Missing values and other possible special values are treated 
separately from so-called complete cases.
This is replica of monobin R package.

## Installation
Currently github and testing versions are available.</br>
To install github version run the following code:
```shell
pip install git+https://github.com/andrija-djurovic/monobinpy.git#egg=monobinpy
```
and to install testing version:
``` 
$ pip install -i https://test.pypi.org/simple/ monobinpy
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
For more examples and package functions check the help page:
```python
help(mb) 
```
