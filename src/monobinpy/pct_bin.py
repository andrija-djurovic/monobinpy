import pandas as pd
import numpy as np
from ._helpers import *

def pct_bin(x, y, 
            sc = [float("NaN"), float("Inf"), float("-Inf")], 
            sc_method = "together", 
            g = 15, 
            y_type = "guess", 
            woe_trend = True, 
            force_trend = "guess"):
    """

    Percentile-based monotonic binning by the iterative discretization.

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
                    woe_trend: Applied only for a continuous target (y) as weights of evidence 
                               (WoE) trend check. 
                               Default is TRUE.
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


    
    if not isinstance(woe_trend, bool):
       raise Error("woe_trend has to be boolean of lenght one.") 
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
    if (len(checks_res[0]) > 0):
        return(eval(checks_res[1]))
    y_check = checks_res[2]
    if y_check == "bina":
       ds = pct_bin_bina(tbl_sc = d_sc.copy(), 
                         tbl_cc = d_cc.copy(), 
                         method = sc_method, 
                         g = g, 
                         force_trend = force_trend)    
    if y_check == "cont":
       ds = pct_bin_cont(tbl_sc = d_sc.copy(), 
                         tbl_cc = d_cc.copy(), 
                         method = sc_method, 
                         g = g, 
                         woe_trend = woe_trend,
                         force_trend = force_trend)      
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