import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportions_ztest
    
#checks
class Error(Exception):
    """Check errors"""
    pass
def checks_init(x, y, sc, sc_method, y_type, force_trend):
    if not len(x) == len(y):
       raise Error("x and y have to be of the same length") 
    scm_opts = ["together", "separately"]
    yt_opts = ["guess", "bina", "cont"]
    ft_opts = ["guess", "i", "d"]
    if sc_method not in scm_opts:
       raise Error("sc_method has to be one of: " + \
                    ", ".join(scm_opts) + ".")      
    if y_type not in yt_opts:
       raise Error("y_type has to be one of: " + \
                     ", ".join(yt_opts) + ".")
    if force_trend not in ft_opts:
       raise Error("force_trend has to be one of: " + \
                    ", ".join(ft_opts) + ".")
    if not all([isinstance(x, (float, int)) for x in sc]):
       raise Error("sc has to have all integer or float elements")  
    if (not isinstance(x, pd.Series)) | (not isinstance(y, pd.Series)):
       raise Error("x and y have to be pandas.Series")
    if not all([isinstance(x, (float, int)) for x in y]):
       raise Error("y has to have all integer or float elements")          
    if not all([isinstance(z, (float, int)) for z in x]):
       raise Error("x has to have all integer or float elements")
    pass
def checks_iter(d, d_cc, y_type):
    cond_01 = len(d_cc.y.unique()) == 1
    cond_02 = len(d_cc.x.unique()) == 1
    if y_type == "guess":
       cond_03 = False
       if all(d_cc.y.isin([0, 1])): 
          y_check = "bina" 
       else: 
        y_check = "cont"
    else:
     if y_type == "bina":
        cond_03 = not d_cc.y.isin([0, 1]).sum() == len(d_cc.y)
        y_check = y_type
     else:
        cond_03 = False
        y_check = y_type  
    cond_04 = d_cc.shape[0] == 0
    cond = [cond_01, cond_02, cond_03, cond_04]
    cond_msg_01 = "pd.DataFrame({'bin' : ['y has a single unique value for the complete cases']})"
    cond_msg_02 = "pd.DataFrame({'bin' : ['x has a single unique value for the complete cases']})"
    cond_msg_03 = "raise Error('y is not 0/1 variable')"
    cond_msg_04 = "pd.DataFrame({'bin' : ['no complete cases']})"
    cond_msg = [cond_msg_01, cond_msg_02, cond_msg_03, cond_msg_04]
    which_cond = np.where(cond)
    if np.size(which_cond):
       first_cond = np.min(which_cond)
       res = [[first_cond], cond_msg[first_cond], y_check]
    else:
       res = [[], [], y_check]
    return(res)

#formatting bins
def format_bin_aux (l, ll, u, bin):
    if abs(l - u) < 1e-8:
       bin_f = bin + " [" + str(round(l, 4)) + " ]"
    elif l == float("-Inf"):
       bin_f = bin + " (" + str(round(l, 4)) + ", " + str(round(ll, 4)) + ")"
    else:
       bin_f = bin + " [" + str(round(l, 4)) + ", " + str(round(ll, 4)) + ")" 
    return(bin_f)
def format_bin(x_lb, x_ub):
    x_lb[0] = float("-Inf") 
    x_lb_lag = x_lb[1:] + [float("Inf")]
    indx = list(range(1, len(x_lb) + 1))
    bin_n = ["%02d"%bin for bin in indx]
    bin_f = list(map(format_bin_aux, x_lb, 
                                     x_lb_lag, 
                                     x_ub, 
                                     bin_n))
    return(bin_f)

#summary table
def tbl_summary(tbl, g_tot, b_tot, n_tot, y_tot, y_type = "bina"):
    gb = tbl.copy().groupby("bin", dropna = False)
    gb_a = gb.aggregate({"y" : [("no", len), 
                                ("y_sum", sum), 
                                ("y_avg", np.average)],
                         "x" : [("x_avg", np.average),
                                ("x_min", min), 
                                ("x_max", max)]})
    gb_a.columns = gb_a.columns.droplevel(0)
    gb_a = gb_a.reset_index()
    gb_a["so"] = g_tot +  b_tot
    
    if y_type == "bina":
       gb_a["sg"] = g_tot
       gb_a["sb"] = b_tot
       gb_a["dist_g"] = (gb_a.no - gb_a.y_sum) / gb_a.sg
       gb_a["dist_b"] = gb_a.y_sum / gb_a.sb
       gb_a["woe"] = np.log(gb_a.dist_g / gb_a.dist_b)
       gb_a["iv_b"] = (gb_a.dist_g - gb_a.dist_b) * gb_a.woe
    else:
       gb_a["sy"] = y_tot
       gb_a["pct_obs"] = gb_a.no / n_tot
       gb_a["pct_y_sum"] = gb_a.y_sum / y_tot
       gb_a["woe"] = np.log(gb_a.pct_y_sum / gb_a.pct_obs)
       gb_a["iv_b"] = (gb_a.pct_y_sum - gb_a.pct_obs) * gb_a.woe
    
    return(gb_a)

#slice variable
def slice_variable(x_orig, x_lb, x_ub, sc_u, sc_g):
    lg = len(x_lb)
    x_trans = x_orig.copy()
    if len(sc_g) > 0:
       if ("SC" in sc_g):
           x_trans[x_trans.isin(sc_u)] = "SC"
    bins = format_bin(x_lb = x_lb,  x_ub = x_ub)
    for i in list(range(0, lg)):
        x_lb_l = x_lb[i]
        x_ub_l = x_ub[i]
        bin_l = bins[i]
        indx = ~x_orig.isin(sc_u) & ~x_orig.isin(list(sc_g)) & \
                x_orig.ge(x_lb_l) & x_orig.le(x_ub_l)
        x_trans.loc[indx] = bin_l
    return(x_trans)

#pct_bin_aux for binary variable
def pct_bin_bina(tbl_sc, tbl_cc, method, g, force_trend):
    y_tot = tbl_sc.shape[0] + tbl_cc.shape[0]
    b_tot = sum(tbl_sc.y) + sum(tbl_cc.y)
    g_tot = y_tot - b_tot
    #special cases
    if tbl_sc.shape[0] > 0:	
       if method == "together":
          tbl_sc["bin"] = "SC"
       else:
          tbl_sc["bin"] = tbl_sc.x.copy()
       tbl_sc_s = tbl_summary(tbl = tbl_sc.copy(), 
                              g_tot = g_tot, 
                              b_tot = b_tot, 
                              n_tot = 0, 
                              y_tot = 0, 
                              y_type = "bina")
       tbl_sc_s["type"] = "special cases"
    else:
       tbl_sc_s = pd.DataFrame()       
    #complete cases
    if force_trend == "i":
       cond_exp = "round(cor_coef, 8) == 1 or g == 1"  
    if force_trend == "d":
       cond_exp = "round(cor_coef, 8) == -1 or g == 1"     
    if force_trend == "guess":
       cond_exp = "round(cor_coef, 8) == 1 or round(cor_coef, 8) == -1 or g == 1"
    while True:
        tbl_cc["bin"] = pd.qcut(x = tbl_cc.x.copy(), 
                                q = g, 
                                duplicates = "drop")
        tbl_cc_s = tbl_summary(tbl = tbl_cc.copy(), 
                               g_tot = g_tot, 
                               b_tot = b_tot, 
                               n_tot = 0, 
                               y_tot = 0, 
                               y_type = "bina")
        if tbl_cc_s.shape[0] == 1: break
        cor_coef, p = spearmanr(a = tbl_cc_s.y_avg.copy(),
                                b = tbl_cc_s.x_avg.copy())
        if eval(cond_exp): break
        g = g - 1      
    tbl_cc_s["bin"] = format_bin(x_lb = tbl_cc_s.x_min.tolist(), 
                                 x_ub = tbl_cc_s.x_max.tolist())
    tbl_cc_s["type"] = "complete cases"
    tbl_s = pd.concat([tbl_sc_s, tbl_cc_s], ignore_index = True)
    return(tbl_s)

#pct_bin_aux for continuous variable
def pct_bin_cont(tbl_sc, tbl_cc, method, g, woe_trend, force_trend):   
    n_tot = tbl_sc.shape[0] + tbl_cc.shape[0]
    y_tot = sum(tbl_sc.y) + sum(tbl_cc.y) 
    #special cases
    if tbl_sc.shape[0] > 0:	
       if method == "together":
          tbl_sc["bin"] = "SC"
       else:
          tbl_sc["bin"] = tbl_sc.x.copy()
       tbl_sc_s = tbl_summary(tbl = tbl_sc.copy(), 
                              g_tot = 0, 
                              b_tot = 0, 
                              n_tot = n_tot, 
                              y_tot = y_tot, 
                              y_type = "cont")
       tbl_sc_s["type"] = "special cases"
    else:
       tbl_sc_s = pd.DataFrame() 
    #complete cases
    if force_trend == "i":
       cond_exp_1 = "round(cor_coef, 8) == 1 or g == 1"
       cond_exp_2 = "all(np.diff(tbl_cc_s.woe) > 0)"  
    if force_trend == "d":
       cond_exp_1 = "round(cor_coef, 8) == -1 or g == 1"  
       cond_exp_2 = "all(np.diff(tbl_cc_s.woe) < 0)"
    if force_trend == "guess":
       cond_exp_1 = "round(cor_coef, 8) == 1 or round(cor_coef, 8) == -1 or g == 1"
       cond_exp_2 = "all(np.diff(tbl_cc_s.woe) > 0) or all(np.diff(tbl_cc_s.woe) < 0)"
    while True:
        tbl_cc["bin"] = pd.qcut(x = tbl_cc.x.copy(), 
                                q = g, 
                                duplicates = "drop")
        tbl_cc_s = tbl_summary(tbl = tbl_cc.copy(), 
                               g_tot = 0, 
                               b_tot = 0, 
                               n_tot = n_tot, 
                               y_tot = y_tot, 
                               y_type = "cont")
        if tbl_cc_s.shape[0] == 1: break
        if woe_trend:
            monocheck = eval(cond_exp_2)
        else:
             cor_coef, p = spearmanr(a = tbl_cc_s.y_avg.copy(),
                                     b = tbl_cc_s.x_avg.copy())
             monocheck = eval(cond_exp_1)
        if monocheck or g == 1: break
        g = g - 1     
    tbl_cc_s["bin"] = format_bin(x_lb = tbl_cc_s.x_min.tolist(), 
                                 x_ub = tbl_cc_s.x_max.tolist())
    tbl_cc_s["type"] = "complete cases"
    tbl_s = pd.concat([tbl_sc_s, tbl_cc_s], ignore_index = True)
    return(tbl_s)

#iso summary
def iso_summary(tbl, bin):
    tbl_g = tbl.groupby(bin, dropna = False)
    tbl_s = tbl_g.aggregate({"y" : [("no", len), 
                                 ("y_sum", sum), 
                                 ("y_avg", np.average)],
                             "x" : [("x_avg", np.average),
                                 ("x_min", min), 
                                 ("x_max", max)]})
    tbl_s.columns = tbl_s.columns.droplevel(0)
    tbl_s = tbl_s.reset_index()
    return(tbl_s)

#woe calculation
def woe_calc(tbl, y_check):
    if y_check == "bina":
       tbl["so"] = sum(tbl.no)
       tbl["sg"] = sum(tbl.no) - sum(tbl.y_sum)
       tbl["sb"] = sum(tbl.y_sum)
       tbl["dist_g"] = (tbl.no - tbl.y_sum) / tbl.sg
       tbl["dist_b"] = tbl.y_sum / tbl.sb
       tbl["woe"] = np.log(tbl.dist_g / tbl.dist_b)
       tbl["iv_b"] = (tbl.dist_g - tbl.dist_b) * tbl.woe
    else:
       tbl["so"] = sum(tbl.no)
       tbl["sy"] = sum(tbl.y_sum)
       tbl["pct_obs"] = tbl.no / tbl.so
       tbl["pct_y_sum"] = tbl.y_sum / tbl.sy
       tbl["woe"] = np.log(tbl.pct_y_sum / tbl.pct_obs)
       tbl["iv_b"] = (tbl.pct_y_sum - tbl.pct_obs) * tbl.woe
    return(tbl)

#correction for min num of obs and target rate
def tbl_correction(tbl, mno, mrate, what, y_check):
    wm = lambda x: np.average(x, weights = tbl.loc[x.index, "no"])
    if what == "obs":
       cn = "no"; thr = mno
    else:
       if y_check == "bina": 
          cn = "y_sum" 
       else: 
          cn = "y_avg"
       thr = mrate
    while True:
       if tbl.shape[0] == 1: break
       values = tbl[cn].copy()
       if all(values >= thr): break
       gap = np.min(np.where(values < thr))
       if gap == (tbl.shape[0] - 1):
          tbl.loc[[gap - 1], ["bin"]] = tbl.loc[[gap]]["bin"].tolist()
       else:
          tbl.loc[[gap + 1], ["bin"]] = tbl.loc[[gap]]["bin"].tolist()
       tbl["y_avg"] = tbl.groupby("bin")["y_avg"].transform(wm).tolist()
       tbl["x_avg"] = tbl.groupby("bin")["x_avg"].transform(wm).tolist()
       tbl = tbl.copy().groupby("bin")
       tbl = tbl.aggregate(no = ("no", sum),
                           y_sum = ("y_sum", sum),
                           y_avg = ("y_avg", np.average),
                           x_avg = ("x_avg", np.average),
                           x_min = ("x_min", min),
                           x_max = ("x_max", max))
       tbl = tbl.reset_index()   
    return(tbl)

#iso binning core
def iso(tbl_sc, tbl_cc, method, min_obs, min_rate, y_check, force_trend):
    #special cases
    if tbl_sc.shape[0] > 0:	
       if method == "together":
          tbl_sc["bin"] = "SC"
       else:
          tbl_sc["bin"] = tbl_sc.x.copy()
       tbl_sc_s = iso_summary(tbl = tbl_sc.copy(), bin = "bin")
       tbl_sc_s["type"] = "special cases"
    else:
       tbl_sc_s = pd.DataFrame() 
    #complete cases
    if force_trend == "guess":
       cor_coef, p = spearmanr(a = tbl_cc.x.copy(),
                               b = tbl_cc.y.copy())
       cc_sign = np.where(cor_coef >= 0, 1, -1)
    else:
       cc_sign = np.where(force_trend == "i", 1, -1)
    ir = IsotonicRegression()
    ir_fit = ir.fit_transform(X = tbl_cc.x.copy(), 
                              y = cc_sign * tbl_cc.y.copy())
    tbl_cc["y_hat"] = ir_fit
    tbl_cc_s = iso_summary(tbl = tbl_cc.copy(), bin = "y_hat")
    tbl_cc_s.sort_values(by = ["y_hat"], 
                         ascending = np.where(cc_sign == -1, False, True).tolist(),
                         inplace = True)
    tbl_cc_s.drop(labels = "y_hat", axis = 1, inplace = True)
    tbl_cc_s.insert(loc = 0, 
                    column = "bin",  
                    value = list(range(0, tbl_cc_s.shape[0])))
    tbl_cc_s = tbl_correction(tbl = tbl_cc_s.copy(), 
                              mno = min_obs, 
                              mrate = min_rate, 
                              what = "obs", 
                              y_check = y_check)
    tbl_cc_s = tbl_correction(tbl = tbl_cc_s.copy(), 
                              mno = min_obs, 
                              mrate = min_rate, 
                              what = "rate", 
                              y_check = y_check)
    tbl_cc_s.sort_values(by = ["x_avg"], inplace = True)
    tbl_cc_s["bin"] = format_bin(x_lb = tbl_cc_s.x_min.tolist(), 
                                 x_ub = tbl_cc_s.x_max.tolist())
    tbl_cc_s["type"] = "complete cases"
    tbl_s = pd.concat([tbl_sc_s, tbl_cc_s], ignore_index = True)
    return(tbl_s)

#stepwise for ndr method
def stepwise_reg(data_lr, nst_d, p_val, y_check):
    if y_check == "bina":
       reg_exp = """smf.glm(formula = reg_f, 
                            data = data_lr, 
                            family = sm.families.Binomial())"""
    else:
        reg_exp = """smf.ols(formula = reg_f, 
                             data = data_lr)"""   
    while True:
        reg_f = "y ~ " + " + ".join(nst_d)
        model = eval(reg_exp)
        lr = model.fit()
        p_vals = lr.pvalues.iloc[1:]
        cond = all(p_vals < p_val)
        if cond: break
        p_max = p_vals.argmax()
        nst_d.pop(p_max)
        if len(nst_d) == 0: break
    return(nst_d)

#ndr method
def ndr(tbl, mdb, p_val,y_check):
    if tbl.shape[0] == 1:
       return(tbl)
    level_u = tbl.x_min.copy()
    for i in list(range(1, len(level_u))):
        level_l = level_u.loc[i].copy()
        nd = np.where(mdb.x < level_l, 0, 1) 
        mdb = pd.concat([mdb, pd.Series(nd)],
                        axis = 1)
        mdb.columns = [*mdb.columns[:-1], "dv_" + str(i)]
    nst_d = [col for col in [*mdb.columns] if "dv_" in col] 
    lrm = stepwise_reg(data_lr = mdb.copy(),
                       nst_d = nst_d,
                       p_val = p_val,
                       y_check = y_check) 
    if len(lrm) > 0:
       indx = [int(x.split("_")[1]) for x in nst_d]
       cp = level_u.iloc[indx].tolist()
       cp = [min(mdb.x)] + cp + [max(mdb.x) + 1]
       bin = pd.cut(x = mdb.x.copy(), 
                    bins = cp, 
                    right = False,
                    include_lowest = True)
       mdb["bin"] = bin
    else:
       mdb["bin"] = "NS" 
    res = iso_summary(tbl = mdb.copy(), bin = "bin")
    res["bin"] = format_bin(x_lb = res.x_min.tolist(),
                            x_ub = res.x_max.tolist())
    res["type"] = "complete cases"
    return(res)

#cum_bin_aux
def cum_bin_aux(tbl, force_trend, y_check):
    if force_trend == "guess":
       cor_coef, p = spearmanr(a = tbl.x.copy(),
                               b = tbl.y.copy())
       sort_d = np.where(cor_coef >= 0, 1, -1)
    else:
       sort_d = np.where(force_trend == "i", 1, -1)
    tbl = iso_summary(tbl = tbl.copy(), bin = "bin")
    if tbl.shape[0] == 1:
       tbl["type"] = "complete cases"
       tbl["bin"] = format_bin(x_lb = tbl.x_min.tolist(),
                               x_ub = tbl.x_max.tolist())
       return(tbl)
    tbl.sort_values(by = ["bin"], 
                    ascending = np.where(sort_d == 1, False, True).tolist(),
                    inplace = True)
    if all(np.diff(tbl.y_avg) > 0):
        tbl["type"] = "complete cases"
        tbl["bin"] = format_bin(x_lb = tbl.x_min.tolist(),
                                x_ub = tbl.x_max.tolist())
        return(tbl) 
    tbl_i = tbl.copy()
    cp = []
    while True:
        cs = tbl_i.y_sum.cumsum() / tbl_i.no.cumsum()
        indx = cs.argmax()
        if indx == (tbl_i.shape[0] - 1):
           cp = cp + [indx + 1]
           break
        cp = cp + [indx + 1]
        tbl_i = tbl_i.iloc[(indx + 1):].copy()
    if len(cp) == 1:
       tbl["bin"] = 1
    else:
       mod = np.repeat(np.cumsum(cp), cp)
       tbl["bin"] = mod
    wm = lambda x: np.average(x, weights = tbl.loc[x.index, "no"])
    tbl["y_avg"] = tbl.groupby("bin")["y_avg"].transform(wm).tolist()
    tbl["x_avg"] = tbl.groupby("bin")["x_avg"].transform(wm).tolist()
    tbl = tbl.groupby("bin")
    tbl_s = tbl.aggregate(no = ("no", sum),
                          y_sum = ("y_sum", sum),
                          y_avg = ("y_avg", np.average),
                          x_avg = ("x_avg", np.average),
                          x_min = ("x_min", min),
                          x_max = ("x_max", max))
    tbl_s = tbl_s.reset_index() 
    tbl_s.sort_values(by = ["x_min"], inplace = True)
    tbl_s["bin"] = format_bin(x_lb = tbl_s.x_min.tolist(), 
                              x_ub = tbl_s.x_max.tolist())
    tbl_s["type"] = "complete cases"
    return(tbl_s)
    
#woe_bin_aux
def woe_bin_aux(tbl, woe_gap, y_check):
    tbl_sc = tbl[tbl.type.isin(["special cases"])].copy()
    tbl_cc = tbl[tbl.type.isin(["complete cases"])].copy()
    tbl_cc.sort_values(by = ["y_avg"], inplace = True)
    tbl_cc["bin"] = ["%02d"%bin for bin in [*range(tbl_cc.shape[0])]]
    woe_gap = np.where(y_check == "bina", -woe_gap, woe_gap).tolist()
    cond = np.where(y_check == "bina", 
                    "all(woe_diff < woe_gap)", 
                    "all(woe_diff > woe_gap)").tolist()
    gap_indx = np.where(y_check == "bina", 
                    "woe_diff.argmax() + 1", 
                    "woe_diff.argmin() + 1").tolist()
    while True:
        if tbl_cc.shape[0] == 1: break
        woe_diff = np.diff(tbl_cc.woe)
        if eval(cond): break
        gap = eval(gap_indx)
        tbl_cc.loc[[gap - 1], ["bin"]] = tbl_cc.loc[[gap]]["bin"].tolist()
        wm = lambda x: np.average(x, weights = tbl_cc.loc[x.index, "no"])
        tbl_cc["y_avg"] = tbl_cc.groupby("bin")["y_avg"].transform(wm).tolist()
        tbl_cc["x_avg"] = tbl_cc.groupby("bin")["x_avg"].transform(wm).tolist()
        tbl_cc_s = tbl_cc.groupby("bin")
        tbl_cc_s = tbl_cc_s.aggregate(no = ("no", sum),
                                    y_sum = ("y_sum", sum),
                                    y_avg = ("y_avg", np.average),
                                    x_avg = ("x_avg", np.average),
                                    x_min = ("x_min", min),
                                    x_max = ("x_max", max))
        tbl_cc_s["type"] = "complete cases"
        tbl_cc_s = tbl_cc_s.reset_index() 
        tbl_cc_s = woe_calc(tbl = pd.concat([tbl_sc[tbl_cc_s.columns],
                                             tbl_cc_s.copy()]), 
                            y_check = y_check)
        tbl_cc = tbl_cc_s[tbl_cc_s.type.isin(["complete cases"])].copy()
    tbl_cc.sort_values(by = ["x_avg"], inplace = True)
    tbl_cc["bin"] = format_bin(x_lb = tbl_cc.x_min.tolist(), 
                               x_ub = tbl_cc.x_max.tolist())
    res = pd.concat([tbl_sc, tbl_cc])
    return(res)
    
#test of 2 proportions - adjacent bins
def t2p_merge(tbl, sig):
    if tbl.shape[0] == 1:
       tbl["p_val"] = float("NaN")
       return(tbl)
    cor_coef, p = spearmanr(a = tbl.y_avg.copy(),
                            b = tbl.x_avg.copy())
    sts = np.where(cor_coef >= 0, "smaller", "larger").tolist()
    test_exp = "proportions_ztest(count = counts, nobs = nobs, alternative = sts)"
    tbl["p_val"] = float("NaN")
    for i in [*range(1, tbl.shape[0])]:
        counts = [tbl.y_sum.iloc[i - 1], tbl.y_sum.iloc[i]]
        nobs = [tbl.no.iloc[i - 1], tbl.no.iloc[i]]
        test_stat, p_value = eval(test_exp)
        tbl.iloc[i, tbl.columns.get_loc("p_val")] = p_value
    tbl["mod"] = [*range(tbl.shape[0])]
    while True:
        if all(tbl.p_val[1:] < sig): break
        if tbl.shape[0] == 1: break
        p_max = tbl.p_val.argmax() 
        tbl.iloc[p_max - 1, 
                 tbl.columns.get_loc("mod")] = tbl.iloc[p_max]["mod"]   
        bm = [tbl.iloc[p_max].bin, tbl.iloc[p_max - 1].bin]
        #find previous and next bins after merging
        if p_max - 2 < 0:
           bp = ""
        else:
           bp = tbl.iloc[p_max - 2].bin 
        if p_max + 1 > (tbl.shape[0] - 1):
           bn = ""
        else:
           bn = tbl.iloc[p_max + 1].bin
        #recalculate p-values for new group and its neighbors
        if bp != "":
           count_1 = tbl.y_sum[tbl.bin.isin([bp])].tolist()[0]
           count_2 = tbl.y_sum[tbl.bin.isin(bm)].sum()
           nob_1 = tbl.no[tbl.bin.isin([bp])].tolist()[0]
           nob_2 = tbl.no[tbl.bin.isin(bm)].sum()
           counts = [count_1, count_2]
           nobs = [nob_1, nob_2]
           test_stat, p_value = eval(test_exp)
           tbl.loc[tbl["mod"].isin([tbl["mod"].loc[p_max]]), 
                   "p_val"] = p_value          
        else:
           p_value = tbl.p_val.iloc[p_max].copy()
           tbl.iloc[p_max - 1, tbl.columns.get_loc("p_val")] = p_value
        if bn != "":
           count_1 = tbl.y_sum[tbl.bin.isin(bm)].sum()
           count_2 = tbl.y_sum[tbl.bin.isin([bn])].tolist()[0]
           nob_1 = tbl.no[tbl.bin.isin(bm)].sum()
           nob_2 = tbl.no[tbl.bin.isin([bn])].tolist()[0]
           counts = [count_1, count_2]
           nobs = [nob_1, nob_2]
           test_stat, p_value = eval(test_exp)
           tbl.iloc[p_max + 1, tbl.columns.get_loc("p_val")] = p_value
        else:
           p_value = tbl.p_val.iloc[p_max].copy()
           tbl.iloc[p_max - 1, tbl.columns.get_loc("p_val")] = p_value   
        if bp == "" and bn == "":
           tbl.iloc[[(p_max - 1), p_max], 
                    tbl.columns.get_loc("p_val")] = 1
              
        wm = lambda x: np.average(x, weights = tbl.loc[x.index, "no"])
        tbl["y_avg"] = tbl.groupby("mod")["y_avg"].transform(wm).tolist()
        tbl["x_avg"] = tbl.groupby("mod")["x_avg"].transform(wm).tolist()
        tbl = tbl.groupby("mod")
        tbl = tbl.aggregate(bin = ("bin", lambda x: " + ".join(x)),
                            no = ("no", sum),
                            y_sum = ("y_sum", sum),
                            y_avg = ("y_avg", np.average),
                            x_avg = ("x_avg", np.average),
                            x_min = ("x_min", min),
                            x_max = ("x_max", max),
                            p_val = ("p_val", np.average))
        tbl = tbl.reset_index() 
        tbl.loc[0, "p_val"] = float("NaN")
    tbl.drop(labels = "mod", axis = 1, inplace = True)
    tbl.sort_values(by = ["x_avg"], inplace = True)
    tbl["bin"] = format_bin(x_lb = tbl.x_min.tolist(), 
                            x_ub = tbl.x_max.tolist())    
    tbl["type"] = "complete cases"
    return(tbl)

#t-test - adjacent bins
def ttg_merge(tbl, sig, ds):
    if tbl.shape[0] == 1:
       tbl["p_val"] = 99
       return(tbl)
    cor_coef, p = spearmanr(a = tbl.y_avg.copy(),
                            b = tbl.x_avg.copy())
    sts = np.where(cor_coef >= 0, "smaller", "larger").tolist()
    test_exp = "sm.stats.ttest_ind(x1 = y_1, x2 = y_2, usevar='unequal', alternative = sts)"
    tbl["p_val"] = float("NaN")
    for i in [*range(1, tbl.shape[0])]:
        y_1 = ds.loc[(ds.x >= tbl.x_min.iloc[i - 1]) & \
                     (ds.x <= tbl.x_max.iloc[i - 1]), "y"].copy()
        y_2 = ds.loc[(ds.x >= tbl.x_min.iloc[i]) & \
                     (ds.x <= tbl.x_max.iloc[i]), "y"].copy()
        test_stat, p_value = eval(test_exp)[:-1]
        tbl.iloc[i, tbl.columns.get_loc("p_val")] = p_value
    tbl["mod"] = [*range(tbl.shape[0])]
    while True:
        if all(tbl.p_val[1:] < sig): break
        if tbl.shape[0] == 1: break
        p_max = tbl.p_val.argmax() 
        tbl.iloc[p_max - 1, 
                 tbl.columns.get_loc("mod")] = tbl.iloc[p_max]["mod"]   
        bm = [tbl.iloc[p_max].bin, tbl.iloc[p_max - 1].bin]
        #find previous and next bins after merging
        if p_max - 2 < 0:
           bp = ""
        else:
           bp = tbl.iloc[p_max - 2].bin 
        if p_max + 1 > (tbl.shape[0] - 1):
           bn = ""
        else:
           bn = tbl.iloc[p_max + 1].bin
        #recalculate p-values for new group and its neighbors
        if bp != "":
           y_1 = ds.loc[(ds.x >= tbl.x_min[tbl.bin.isin([bp])].tolist()[0]) & \
                        (ds.x <= tbl.x_max[tbl.bin.isin([bp])].tolist()[0]), \
                        "y"].copy()
           y_2 = ds.loc[(ds.x >= min(tbl.x_min[tbl.bin.isin(bm)])) & \
                        (ds.x <= max(tbl.x_max[tbl.bin.isin(bm)])), \
                        "y"].copy()
           test_stat, p_value = eval(test_exp)[:-1]
           tbl.loc[tbl["mod"].isin([tbl["mod"].loc[p_max]]), 
                   "p_val"] = p_value          
        else:
           p_value = tbl.p_val.iloc[p_max].copy()
           tbl.iloc[p_max - 1, tbl.columns.get_loc("p_val")] = p_value
        if bn != "":
           y_1 = ds.loc[(ds.x >= min(tbl.x_min[tbl.bin.isin(bm)])) & \
                         (ds.x <= max(tbl.x_max[tbl.bin.isin(bm)])), \
                         "y"].copy()
           y_2 = ds.loc[(ds.x >= tbl.x_min[tbl.bin.isin([bn])].tolist()[0]) & \
                         (ds.x <= tbl.x_max[tbl.bin.isin([bn])].tolist()[0]), \
                             "y"].copy() 
           test_stat, p_value = eval(test_exp)[:-1]
           tbl.iloc[p_max + 1, tbl.columns.get_loc("p_val")] = p_value
        else:
           p_value = tbl.p_val.iloc[p_max].copy()
           tbl.iloc[p_max - 1, tbl.columns.get_loc("p_val")] = p_value   
        if bp == "" and bn == "":
           tbl.iloc[[(p_max - 1), p_max], 
                    tbl.columns.get_loc("p_val")] = 1
              
        wm = lambda x: np.average(x, weights = tbl.loc[x.index, "no"])
        tbl["y_avg"] = tbl.groupby("mod")["y_avg"].transform(wm).tolist()
        tbl["x_avg"] = tbl.groupby("mod")["x_avg"].transform(wm).tolist()
        tbl = tbl.groupby("mod")
        tbl = tbl.aggregate(bin = ("bin", lambda x: " + ".join(x)),
                            no = ("no", sum),
                            y_sum = ("y_sum", sum),
                            y_avg = ("y_avg", np.average),
                            x_avg = ("x_avg", np.average),
                            x_min = ("x_min", min),
                            x_max = ("x_max", max),
                            p_val = ("p_val", np.average))
        tbl = tbl.reset_index() 
        tbl.loc[0, "p_val"] = float("NaN")
    tbl.drop(labels = "mod", axis = 1, inplace = True)
    tbl.sort_values(by = ["x_avg"], inplace = True)
    tbl["bin"] = format_bin(x_lb = tbl.x_min.tolist(), 
                            x_ub = tbl.x_max.tolist())    
    tbl["type"] = "complete cases"
    return(tbl)
