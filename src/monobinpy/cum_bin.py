import pandas as pd
import numpy as np
from ._helpers import *

def cum_bin(x, y, 
            sc = [float("NaN"), float("Inf"), float("-Inf")], 
            sc_method = "together", 
            g = 15, 
            y_type = "guess",  
            force_trend = "guess"):
    
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
