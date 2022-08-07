import pandas as pd
import numpy as np
from ._helpers import *

def iso_bin(x, y, 
            sc = [float("NaN"), float("Inf"), float("-Inf")], 
            sc_method = "together", 
            y_type = "guess", 
            min_pct_obs = 0.05,
            min_avg_rate = 0.01, 
            force_trend = "guess"):
    
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



    
    
    