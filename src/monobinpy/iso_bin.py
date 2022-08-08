import pandas as pd
import numpy as np
from ._helpers import *

def iso_bin(x, y, 
            sc = [float("NaN"), float("Inf"), float("-Inf")], 
            sc_method = "together", 
            min_pct_obs = 0.05,
            min_avg_rate = 0.01, 
            y_type = "guess", 
            force_trend = "guess"):
    """

    Three-stage monotonic binning procedure. The first stage is isotonic regression used 
    to achieve the monotonicity, while the remaining two stages are possible corrections for
    minimum percentage of observations and target rate.

            Parameters:
            ------------
                    x: Pandas series to be binned.
                    y: Pandas series - target vector (binary or continuous).
                    sc: List with special case elements. 
                        Default values are [float("NaN"), float("Inf"), float("-Inf")].
                        Recommendation is to keep the default values always and add new ones if needed. 
                        Otherwise, if these values exis in x and are not defined in the sc list, 
                        function will report the error.  
                    sc_method: Define how special cases will be treated, all together or in separate bins.
                               Possible values are 'together' (default), 'separately'.
                    min_pct_obs: Minimum percentage of observations per bin. 
                                 Default is 0.05 or minimum 30 observations.
                    min_avg_rate: Minimum y average rate. 
                                  Default is 0.01 or minimum 1 bad case for y 0/1.
                    y_type: Type of y, possible options are 'bina' (binary), 'cont' (continuous) and 'guess'.
                           If default value - 'guess' is passed, then algorithm will identify if y is 
                           0/1 or continuous variable.
                    force_trend: If the expected trend should be forced. 
                                 Possible values: 'i' for increasing trend 
                                 (y increases with increase of x), 'd' for decreasing trend 
                                 (y decreases with decrease of x) and 'guess'. Default value is 'guess'. 
                                 If the default value is passed, then trend will be identified 
                                 based on the sign of the Spearman correlation coefficient 
                                 between x and y on complete cases.

            Returns:
            ------------
                    List of two objects. The first object, pandas data frame presents 
                    a summary table of final binning, while second one is a pandas series 
                    of discretized values. In case of single unique value for x or y 
                    in complete cases (cases different than special cases), 
                    it will return data frame with info.
    """    

    checks_init(x = x.copy(),
                y = y.copy(),
                sc = sc,
                sc_method = sc_method, 
                y_type = y_type, 
                force_trend = force_trend)
    d = pd.DataFrame({"y" : y, "x" : x})
    d_sc = d[d.x.isin(sc)] 
    d_cc = d[~d.x.isin(sc)] 
    checks_res = checks_iter(d = d, 
                             d_cc = d_cc, 
                             y_type = y_type)
    if len(checks_res[0]) > 0:
       return(eval(checks_res[1]))
    y_check = checks_res[2]
    nr = d.shape[0]
    min_obs = np.ceil(np.where(nr * min_pct_obs < 30, 30, nr * min_pct_obs))
    if y_check == "bina":
       nd = sum(d.y)   
       min_rate = np.ceil(np.where(nd * min_avg_rate < 1, 1, nd * min_avg_rate))
    else:
       min_rate = min_avg_rate 
     
    ds = iso(tbl_sc = d_sc.copy(), 
             tbl_cc = d_cc.copy(), 
             method = sc_method,
             min_obs = min_obs, 
             min_rate = min_rate, 
             y_check = y_check, 
             force_trend = force_trend)
    ds = woe_calc(tbl = ds.copy(), y_check = y_check)

    sc_u = np.unique(sc).tolist()
    sc_g = ds.bin[ds.type.isin(["special cases"])].tolist()
    x_mins = ds.x_min[~ds.bin.isin(sc_u) & ~ds.bin.isin(["SC"])].tolist()
    x_maxs = ds.x_max[~ds.bin.isin(sc_u) & ~ds.bin.isin(["SC"])].tolist()
    x_trans = slice_variable(x_orig = x.copy(), 
                             x_lb = x_mins, 
                             x_ub = x_maxs, 
                             sc_u = sc_u, 
                             sc_g = sc_g)
    return([ds, x_trans])



    
    
    