import pandas as pd
import numpy as np
from ._helpers import *

def cum_bin(x, y, 
            sc = [float("NaN"), float("Inf"), float("-Inf")], 
            sc_method = "together", 
            g = 15, 
            y_type = "guess",  
            force_trend = "guess"):
    """

    Monotonic binning based on maximum cumulative target rate (MAPA).

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
                    g: Number of starting groups. Default is 15.
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
    d_sc = d[d.x.isin(sc)].copy()
    d_cc = d[~d.x.isin(sc)].copy() 
    checks_res = checks_iter(d = d, 
                             d_cc = d_cc, 
                             y_type = y_type)
    if (len(checks_res[0]) > 0):
        return(eval(checks_res[1]))
    y_check = checks_res[2]
    
    #special cases
    if d_sc.shape[0] > 0:	
       if sc_method == "together":
          d_sc["bin"] = "SC"
       else:
          d_sc["bin"] = d_sc.x.copy()
       d_sc_s = iso_summary(tbl = d_sc.copy(), bin = "bin")
       d_sc_s["type"] = "special cases"
    else:
       d_sc_s = pd.DataFrame() 
    #complete cases
    d_cc["bin"] = pd.qcut(x = d_cc.x.copy(), 
                          q = g, 
                          duplicates = "drop")
    d_cc["bin"] = pd.Categorical(d_cc.bin).remove_unused_categories()
    d_cc_s = cum_bin_aux(tbl = d_cc.copy(), 
                         force_trend = force_trend, 
                         y_check = y_check)
    ds = pd.concat([d_sc_s, d_cc_s], ignore_index = True)
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
