import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as spi
import pandas as pd
import glob
import seaborn as sns

def loading_info(info_summary, f_eD1):
    # unpack reference specialist info (f_eD1 = 1)
    i_eD1_spec = np.where(~np.isnan(info_summary.iloc[:,1]))[0][-1]
    Y_eD1_MET_X1 = info_summary.iloc[i_eD1_spec, 3]
    mu_max_X_eD1 = info_summary.iloc[i_eD1_spec, 1]
    q_ecat_max_X_eD1 = info_summary.iloc[i_eD1_spec, -2]
    Y_S_AN_tot = info_summary.iloc[i_eD1_spec, 5]
    Y_S_CAT_tot = Y_eD1_MET_X1 - Y_S_AN_tot

    # unpack other specialist info (f_eD1 = 0)
    i_eD2_spec = np.where(~np.isnan(info_summary.iloc[:,1]))[0][0]
    Y_eD2_MET_X2 = info_summary.iloc[i_eD2_spec, 4]
    mu_max_X_eD2 = info_summary.iloc[i_eD2_spec, 1]

    # unpack generalist info
    # get index corresponding to specified fraction
    f_i = np.argmin(np.abs(info_summary.iloc[:,0] - f_eD1))

    # get total substrate consumed per X
    Y_eD1_MET_X_gen = info_summary.iloc[f_i, 3]
    Y_eD2_MET_X_gen = info_summary.iloc[f_i, 4]

    # get amount of substrate dedicated to anabolism
    Y_S_AN_eD1_gen_tot = info_summary.iloc[f_i, 5]
    Y_S_AN_eD2_gen_tot = info_summary.iloc[f_i, 6]

    # get maximum catabolic consumption rate
    q_ecat_max_X_eD1_gen = info_summary.iloc[f_i, -2]
    q_ecat_max_X_eD2_gen = info_summary.iloc[f_i, -1]

    # get maximum growth rate
    mu_max_fi_gen = info_summary.iloc[f_i, 1]

    return Y_eD1_MET_X1, mu_max_X_eD1, q_ecat_max_X_eD1, Y_S_AN_tot, Y_S_CAT_tot, Y_eD2_MET_X2, mu_max_X_eD2, Y_eD1_MET_X_gen, Y_eD2_MET_X_gen, Y_S_AN_eD1_gen_tot, Y_S_AN_eD2_gen_tot, q_ecat_max_X_eD1_gen, q_ecat_max_X_eD2_gen, mu_max_fi_gen

def fit_kinetics(q_s_max, K_eD1_spec, Y_S_AN_tot, Y_S_CAT_tot, q_s_cat_max, mu_max_X_eD1, eD1_name, NoE):

    kin_figs = []

    # calculating substrate consumption with regular Monod   
    all_eD1_conc_M = np.arange(0, 0.03, 0.000001)   # M 

    # substrate consumption - regular Monod
    q_S = q_s_max * (all_eD1_conc_M / (all_eD1_conc_M + K_eD1_spec))

    # function to fit
    def qs_fit_on_cat(CeD1, Ke):
        return ( 1 + ( Y_S_AN_tot / (Y_S_CAT_tot)) ) * q_s_cat_max * (CeD1 / (CeD1 + Ke )) 

    # initial guesses for Ke
    initial_guess = [100*10**(-2)]

    # fit the model
    popt, _ = curve_fit(qs_fit_on_cat, all_eD1_conc_M , q_S, p0 = initial_guess, bounds=(0, np.inf))
    Ke_fit = popt[0]

    # calculate the substrate consumption rate using the fitted affinity constant
    qS_fit = qs_fit_on_cat(all_eD1_conc_M, *popt)

    # plot
    fig = plt.figure()
    plt.plot( all_eD1_conc_M*1000, -q_S, 'o', label=f'Regular Monod, $K_s$ = {K_eD1_spec*1000:.2e} mM', markersize=4)
    plt.plot( all_eD1_conc_M*1000, -qS_fit, '-', label=f'Fit: $K_e$ = {Ke_fit*1000:.2e} mmol glu./L')
    plt.xlabel(f'{eD1_name} [mM]')
    plt.ylabel('|$q_S$| [molS/Cmol X/h]')
    plt.grid('True')
    plt.legend()
    plt.title(f'Specific substrate consumption rate vs. {eD1_name} concentration \n Fitting the electron affinity')
    plt.close()
    kin_figs += [fig]

    # maximum anabolic substrate uptake rate
    q_s_an_max = Y_S_AN_tot * mu_max_X_eD1

    # function to fit Kc for anabolism
    def qs_fit_on_an(CeD1, Kc):
        return q_s_an_max * (CeD1 / (CeD1 + Kc )) + q_s_cat_max * (CeD1 / (CeD1 + Ke_fit )) 

    # fit the model again against regular Monod (q_S)
    popt, pcov = curve_fit(qs_fit_on_an, all_eD1_conc_M , q_S, p0 = initial_guess, bounds=(0, np.inf))
    Kc_fit = popt[0]

    # catabolism and anabolism separately
    qs_cat = q_s_cat_max * (all_eD1_conc_M / (all_eD1_conc_M + Ke_fit )) 
    qs_an = q_s_an_max * (all_eD1_conc_M / (all_eD1_conc_M + Kc_fit )) 
    qs_tot_fit = qs_cat + qs_an

    # plot
    fig = plt.figure()
    plt.plot( all_eD1_conc_M*1000, -q_S, 'o', label=f'Regular Monod, $K_s$ = {K_eD1_spec*1000:.2e} mM', markersize=4)
    plt.plot( all_eD1_conc_M*1000, -qs_tot_fit, '-', label=f'Sum of anabolic and catabolic fit')
    plt.plot(all_eD1_conc_M*1000, -qs_cat, '--', label=f'Catabolic uptake, $K_e$ = {Ke_fit:.2e} mol/L')
    plt.plot(all_eD1_conc_M*1000, -qs_an, '--', label=f'Anabolic uptake, $K_c$ = {Kc_fit:.2e} mol/L')
    plt.xlabel(f'{eD1_name} [mM]')
    plt.ylabel('|$q_S$| [molS/Cmol X/h]')
    plt.grid('True')
    plt.legend()
    plt.title(f'Specific substrate consumption rate vs. {eD1_name} concentration \n Fitting the electron affinity')
    plt.close()
    kin_figs += [fig]

    return Kc_fit, Ke_fit, kin_figs

def fit_gen_to_spec(Ke_fit, Kc_fit, NoE_eD1, NoE_eD2, eD1_name, eD2_name, f_eD1, q_s_max, K_eD1_spec, Y_eD2_MET_X_gen, Y_eD1_MET_X_gen, Y_eD1_MET_X1, q_ecat_max_X_eD1, q_ecat_max_X_eD1_gen, q_ecat_max_X_eD2_gen, mu_max_fi_gen, Y_S_AN_eD2_gen_tot, Y_S_AN_eD1_gen_tot, kin_figs):    
    # electron substrate concentration
    e_tot = np.arange(0, 0.05, 0.000001)   # milli emol/L

    # convert the fitted catabolic eD1 affinity constant to an electron concentration
    K_e = Ke_fit * NoE_eD1   

    # construct the catabolic consumption uptake curve for the eD1 specialist
    q_e_cat_spec = q_ecat_max_X_eD1 * ( e_tot / (K_e + e_tot) )

    fig = plt.figure()
    plt.plot( e_tot*1000, -q_e_cat_spec, 'o', label=f'|$q_{{cat,max}}$| = {-q_ecat_max_X_eD1:.2f} emol/Cmol X/h, $K_e$ = {K_e*1000:.2e} memol/L', markersize=2)
    plt.xlabel('Substrate [milli emol/L]')
    plt.ylabel('|$q_{e,cat}$| [emol/Cmol X/h]')
    plt.grid('True')
    plt.legend()
    plt.title(f'Specific substrate catabolic consumption rate of {eD1_name} specialist')
    plt.close()
    kin_figs += [fig]


     # function to determine Ke_tot or calculate the substrate uptake rate
    def fit_Ketot_dual_cat(e_tot_fit, Ke_tot):
        return q_ecat_max_X_eD1_gen * ( (f_eD1 * e_tot_fit) / ( (f_eD1 * e_tot_fit) + Ke_tot) ) + q_ecat_max_X_eD2_gen *  ( ( (1 - f_eD1) * e_tot_fit) / ( ( (1 - f_eD1) * e_tot_fit) + Ke_tot) )

    # initial guesses for Ke_tot
    initial_guess = [100*10**(-6)]

    # fit the model
    popt, pcov = curve_fit(fit_Ketot_dual_cat, e_tot, q_e_cat_spec, p0 = initial_guess, bounds=(0, np.inf))
    Ke_tot_fit = popt[0]

    # get values
    qe_cat_fit_gen = fit_Ketot_dual_cat(e_tot, *popt)

    # plot
    fig = plt.figure()
    plt.plot( e_tot*1000, -q_e_cat_spec, 'o', label=f' {eD1_name} specialist, \n |$q_{{cat,max}}$| = {-q_ecat_max_X_eD1:.2f} emol/Cmol X/h, \n $K_e$ = {K_e:.2e} emol/L', markersize=2)
    plt.plot( e_tot*1000, -qe_cat_fit_gen, '-', label=f'Generalist, |$q_{{cat,max,eD1}}$| = {-q_ecat_max_X_eD1_gen:.2f}, \n |$q_{{cat,max,eD2}}$| = {-q_ecat_max_X_eD2_gen:.2f} emol/Cmol X/h, \n $K_{{e,tot}}$ = {Ke_tot_fit:.2e} emol/L')
    plt.xlabel('Substrate [milli emol/L]')
    plt.ylabel('|$q_{e,cat}$| [emol/Cmol X/h]')
    plt.grid('True')
    plt.legend()
    plt.title('Specific substrate catabolic consumption rate of specialist and generalist')
    plt.close()
    kin_figs += [fig]

    # generalist substrate consumption, using the previously determined Ke_tot_fit
    q_cat_eD1_gen = q_ecat_max_X_eD1_gen * ( (f_eD1 * e_tot) / ( (f_eD1 * e_tot) + Ke_tot_fit) )
    q_cat_eD2_gen = q_ecat_max_X_eD2_gen *  ( ( (1 - f_eD1) * e_tot) / ( ( (1 - f_eD1) * e_tot) + Ke_tot_fit) )
    q_cat_gen_tot = q_cat_eD1_gen + q_cat_eD2_gen

    # specialist substrate consumption
    q_cat_eD1_spec = q_ecat_max_X_eD1 * ( (f_eD1 * e_tot) / ( (f_eD1 * e_tot) + K_e) )

    # plot
    fig = plt.figure()
    plt.plot( e_tot*1000, -q_cat_eD1_spec, '-', label=f'{eD1_name} specialist')
    plt.plot( e_tot*1000, -q_cat_gen_tot, '-', color = 'orange', label=f'Generalist')
    plt.plot( e_tot*1000, -q_cat_eD2_gen, '--', color = 'orange', alpha = 0.4, label=f'Generalist, {eD2_name} ')
    plt.plot( e_tot*1000, -q_cat_eD1_gen, '-.', color = 'orange', alpha = 0.6, label=f'Generalist, {eD1_name}')
    plt.xlabel('Substrate [milli emol/L]')
    plt.ylabel('|$q_{e,cat}$| [emol/Cmol X/h]')
    plt.grid('True')
    plt.legend()
    plt.title(f'Specific substrate catabolic consumption rate against total electron availability \n {eD1_name} fraction = {f_eD1:.2f}')
    plt.close()
    kin_figs += [fig]

    Kc_emol = Kc_fit * NoE_eD1 # emol/L

    #substrate uptake from the generalist
    q_eD2_gen = Y_S_AN_eD2_gen_tot * mu_max_fi_gen * ( ( (1 - f_eD1) * e_tot) / ( ( (1 - f_eD1) * e_tot) + Kc_emol) ) + ( q_ecat_max_X_eD2_gen *  ( ( (1 - f_eD1) * e_tot) / ( ( (1 - f_eD1) * e_tot) + Ke_tot_fit) ) ) / NoE_eD2
    q_eD1_gen = Y_S_AN_eD1_gen_tot * mu_max_fi_gen * ( (  f_eD1 * e_tot) / (f_eD1 * e_tot + Kc_emol) ) + ( q_ecat_max_X_eD1_gen *  ( (  f_eD1 * e_tot) / ( (  f_eD1 * e_tot) + Ke_tot_fit) ) ) / NoE_eD1

    # eD1 concentration in molar
    eD1_conc = f_eD1 * e_tot / NoE_eD1
    q_eD1_spec = q_s_max * (eD1_conc / (eD1_conc + K_eD1_spec))

    # plot
    fig = plt.figure()
    plt.plot( e_tot*1000, -q_eD1_spec, '-', label=f' {eD1_name} specialist')
    plt.plot( e_tot*1000, -q_eD2_gen, '--', color = 'orange', alpha = 0.8, label=f'Generalist, {eD2_name} ')
    plt.plot( e_tot*1000, -q_eD1_gen, '-.', color = 'orange', alpha = 0.8, label=f'Generalist, {eD1_name}')
    plt.xlabel('Substrate [milli emol/L]')
    plt.ylabel('|$q_{eDi,tot}$| [emol/Cmol X/h]')
    plt.grid('True')
    plt.legend()
    plt.title(f'Specific substrate consumption rate against total electron availability \n {eD1_name} fraction = {f_eD1:.2f}')
    plt.close()
    kin_figs += [fig]

    fig = plt.figure()
    plt.plot( e_tot*1000, q_eD1_spec/Y_eD1_MET_X1, '-', label=f' {eD1_name} specialist')
    plt.plot( e_tot*1000, q_eD2_gen/Y_eD2_MET_X_gen, '--', color = 'orange', alpha = 0.8, label=f'Generalist, {eD2_name} ')
    plt.plot( e_tot*1000, q_eD1_gen/Y_eD1_MET_X_gen, '-.', color = 'orange', alpha = 0.8, label=f'Generalist, {eD1_name}')
    plt.xlabel('Substrate [milli emol/L]')
    plt.ylabel(' $\mu$ [1/h]')
    plt.grid('True')
    plt.legend()
    plt.title(f'Growth rate against total electron availability \n {eD1_name} fraction = {f_eD1:.2f}')
    plt.close()
    kin_figs += [fig]

    fig = plt.figure()
    plt.plot( e_tot*1000, q_eD1_spec/Y_eD1_MET_X1, '-', label=f' {eD1_name} specialist')
    plt.plot( e_tot*1000, q_eD2_gen/Y_eD2_MET_X_gen, '--', color = 'orange', alpha = 0.8, label=f'Generalist, {eD2_name} ')
    plt.plot( e_tot*1000, q_eD1_gen/Y_eD1_MET_X_gen, '-.', color = 'orange', alpha = 0.8, label=f'Generalist, {eD1_name}')
    plt.xlabel('Substrate [milli emol/L]')
    plt.ylabel(' $\mu$ [1/h]')
    plt.grid('True')
    plt.xlim((0,5))
    plt.legend()
    plt.title(f'Growth rate against total electron availability [zoomed in] \n {eD1_name} fraction = {f_eD1:.2f}')
    plt.close()
    kin_figs += [fig]

    return Kc_emol, Ke_tot_fit, kin_figs

def simulate_competition(excel_file, eD1_info, eD2_info, K_eD1_spec, K_eD2_spec, f_eD1, f_feed, D, e_tot_IN, sim_nHRT, dt = 0.1, c0=[], K_eD1_gen = None, show_kin_figures = False, comp_plots = True):

    # unpack substrate info
    eD1_name, _, NoE_eD1 = eD1_info
    eD2_name, _, NoE_eD2 = eD2_info

    # calculate hydraulic retention time (HRT) from dilution rate (=mu)
    HRT = 1/D

    # retrieve Excel containing stoichiometric info
    info_sum = pd.read_excel(excel_file)
    # load main stoichiometric parameters
    Y_eD1_MET_X1, mu_max_X_eD1, q_ecat_max_X_eD1, Y_S_AN_tot, Y_S_CAT_tot, Y_eD2_MET_X2, mu_max_X_eD2, Y_eD1_MET_X_gen, Y_eD2_MET_X_gen, Y_S_AN_eD1_gen_tot, Y_S_AN_eD2_gen_tot, q_ecat_max_X_eD1_gen, q_ecat_max_X_eD2_gen, mu_max_fi_gen = loading_info(info_sum, f_eD1)

    # convert substrate consumption to mol S/Cmol X/h from emol/Cmol X/h
    q_s_cat_max = -2 / NoE_eD1        # mol S/Cmol X/h
    q_s_max = mu_max_X_eD1 * Y_eD1_MET_X1           # mol S/Cmol X/h

    # fitting electron and carbon affinity of growth on glucose alone
    if K_eD1_gen == None:
        Kc_fit, Ke_fit, kin_figs = fit_kinetics(q_s_max, K_eD1_spec, Y_S_AN_tot, Y_S_CAT_tot, q_s_cat_max, mu_max_X_eD1, eD1_name, NoE_eD1)
    else:
        Kc_fit, Ke_fit, kin_figs = fit_kinetics(q_s_max, K_eD1_gen, Y_S_AN_tot, Y_S_CAT_tot, q_s_cat_max, mu_max_X_eD1, eD1_name, NoE_eD1)
    
    # fitting the generalist mixed growth against growth on glucose alone
    Kc_emol, Ke_tot_fit, kin_figs = fit_gen_to_spec(Ke_fit, Kc_fit, NoE_eD1, NoE_eD2, eD1_name, eD2_name, f_eD1, q_s_max, K_eD1_spec, Y_eD2_MET_X_gen, Y_eD1_MET_X_gen, Y_eD1_MET_X1, q_ecat_max_X_eD1, q_ecat_max_X_eD1_gen, q_ecat_max_X_eD2_gen, mu_max_fi_gen, Y_S_AN_eD2_gen_tot, Y_S_AN_eD1_gen_tot, kin_figs)

    # constants
    MW_X 	= 24 			# g/mol

    # ode-system for the chemostat simulation
    def dcdt (t, c, K_eD1, K_eD2, K_c, K_e, D, e_tot_IN, f):
        # 3 types of biomass (X): eD1 consumers (#1), eD2 consumers (#2) and dual substrate consumers (#3, gen)
        # unpack variables from input
        c_eD1, c_eD2, c_X_eD1, c_X_eD2, c_X_gen = c
        
        # concentrations cannot be negative
        if c_X_eD1 < 1e-12: 
            c_X_eD1 = 0 
        if c_X_eD2 < 1e-12: 
            c_X_eD2 = 0 
        if c_X_gen < 1e-12: 
            c_X_gen = 0

        # determine specific consumption rate for the specialists
        q_eD1_X_eD1 	= mu_max_X_eD1 * Y_eD1_MET_X1 * (c_eD1/(c_eD1 + K_eD1))
        q_eD2_X_eD2 	= mu_max_X_eD2 * Y_eD2_MET_X2 * (c_eD2/(c_eD2 + K_eD2))

        # with the fitted growth curve: do ode-system
        # growth rates
        mu_X_eD1		= q_eD1_X_eD1 / Y_eD1_MET_X1  #1/h
        mu_X_eD2		= q_eD2_X_eD2 / Y_eD2_MET_X2  #1/h

        # growth rate for dual substrate consumption; electron and carbon donor
        q_eD1_X_gen = Y_S_AN_eD1_gen_tot * mu_max_fi_gen * ( (c_eD1 * NoE_eD1) / (c_eD1 * NoE_eD1 + K_c) ) + ( ( q_ecat_max_X_eD1_gen *  ( (c_eD1 * NoE_eD1) / ( c_eD1 * NoE_eD1 + K_e) ) ) / NoE_eD1 )
        q_eD2_X_gen = Y_S_AN_eD2_gen_tot * mu_max_fi_gen * ( (c_eD2 * NoE_eD2) / (c_eD2 * NoE_eD2 + K_c) ) + ( ( q_ecat_max_X_eD2_gen *  ( (c_eD2 * NoE_eD2) / ( c_eD2 * NoE_eD2 + K_e) ) ) / NoE_eD2 )
        
        # calculate the growth rates that could be reached on these substrates - if the other substrate would be consumed in the correct ratio
        mu_gen_on_eD1 = q_eD1_X_gen / Y_eD1_MET_X_gen       # 1/h
        mu_gen_on_eD2 = q_eD2_X_gen / Y_eD2_MET_X_gen       # 1/h

        # pick the limiting qs to determine the growth rate
        # if the growth rate on eD1 is the lowest
        if mu_gen_on_eD1 <= mu_gen_on_eD2:
            # this will be the growth rate
            mu_X_gen = mu_gen_on_eD1                        # 1/h
            # which means q_eD2_X_gen needs to be stoichiometrically adjusted
            q_eD2_X_gen = mu_X_gen * Y_eD2_MET_X_gen        # mol eD2/mol X/h
        # then eD2 consumption is limiting the growth rate
        else:
            mu_X_gen = mu_gen_on_eD2                        # 1/h
            # which means q_eD1_X_gen needs to be stoichiometrically adjusted
            q_eD1_X_gen = mu_X_gen * Y_eD1_MET_X_gen        # mol eD1/mol X/h

        ### ODES ###
        ## LIQUID PHASE ##
        # electron concentration per substrate
        e_eD1_IN = f * e_tot_IN
        e_eD2_IN = (1-f) * e_tot_IN

        # convert electron concentration to molar concentration
        c_eD1_IN = e_eD1_IN / NoE_eD1 
        c_eD2_IN = e_eD2_IN / NoE_eD2 
        
        # substrate change over time
        dceD1_dt = D * (c_eD1_IN - c_eD1) + q_eD1_X_eD1 * c_X_eD1 + q_eD1_X_gen * c_X_gen
        dceD2_dt = D * (c_eD2_IN - c_eD2) + q_eD2_X_eD2 * c_X_eD2 + q_eD2_X_gen * c_X_gen
        
        # 3 types of biomass (1/2/3) &  no biomass coming in!
        dcx1_dt  = D * ( 0 - c_X_eD1) + mu_X_eD1 * c_X_eD1
        dcx2_dt  = D * ( 0 - c_X_eD2) + mu_X_eD2 * c_X_eD2
        dcx3_dt  = D * ( 0 - c_X_gen) + mu_X_gen * c_X_gen
        
        # create array of all compounds to be returned
        dcdt_array = np.array([dceD1_dt, dceD2_dt, dcx1_dt, dcx2_dt, dcx3_dt])
    
        return dcdt_array

    # initial values, if not provided
    if not c0:
        c_glu_0  = f_feed * e_tot_IN / NoE_eD1             #M
        c_ac_0  = (1 - f_feed) * e_tot_IN / NoE_eD2       #M
        c_X0     = 0.1       #g/L

        # assuming equal starting distribution between different types of biomass
        c_X1_0   = (1/3)*c_X0/MW_X    #M
        c_X2_0   = (1/3)*c_X0/MW_X    #M
        c_X3_0   = (1/3)*c_X0/MW_X    #M

        # group initial values
        c0 			= [c_glu_0, c_ac_0, c_X1_0, c_X2_0, c_X3_0]

    # time: initial, final, step
    t0, te      = 0, HRT*sim_nHRT     
    tout 		= np.arange(t0, te, dt)

    print(f'Specialist eD1 = {K_eD1_spec} M')
    print(f'Specialist eD2 = {K_eD2_spec} M')
    print(f'Generalist: $K_{{glu}}$ = {K_eD1_gen}, K_c = {Kc_emol} emol/L, K_e = {Ke_tot_fit} emol/L')
    
    # solve system of equations
    try:
        sol 	= spi.solve_ivp(dcdt, [t0, te], c0, t_eval = tout, method = "Radau", max_step = 0.001, rtol=1e-8, atol=1e-10, 
                                args=(K_eD1_spec, K_eD2_spec, Kc_emol, Ke_tot_fit, D, e_tot_IN, f_feed))
    except:
        print('System could not be solved.')

    # check if the result is valid
    if sol.success == False:
        print(f'No solution was found: "{sol.message}"')
    
    # if so: unpack solutions
    else:
        # check if the solution needs to be downsampled, to avoid that the code crashes during plotting due to a computer memory error!
        # --- Maximum points to keep ---
        max_points = 50000

        # --- Determine downsampling factor ---
        n_points = len(sol.t)
        factor = max(1, n_points // max_points)  # downsample only if needed

        # --- Downsample time and solution arrays in-place ---
        sol.t = sol.t[::factor]
        sol.y = sol.y[:, ::factor]  # keep all variables, downsample along time axis

        t 								            = sol.t
        c_eD1, c_eD2, c_X_eD1, c_X_eD2, c_X_eD3     = sol.y

        c_Xtot  = c_X_eD1 + c_X_eD2 + c_X_eD3
        f_X1    = c_X_eD1/c_Xtot
        f_X2    = c_X_eD2/c_Xtot
        f_X3    = c_X_eD3/c_Xtot

        if comp_plots == True:
            # plot concentrations over time
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18,6))

            # Plot on the left subplot with a secondary y-axis
            ax1 = axes[0]
            ax2 = ax1.twinx()  # Create a secondary y-axis sharing the same x-axis

            # Plot the two variables
            ln1 = ax1.plot(t, c_eD1 * 1000, '-r', label='eD1 [mM]')
            ln2 = ax2.plot(t, c_eD2 * 1000, '-b', label='eD2 [mM]')

            # Color the axes
            ax1.set_ylabel('eD1 [mM]', color='red')
            ax1.tick_params(axis='y', colors='red')
            ax1.spines['left'].set_color('red')

            ax2.set_ylabel('eD2 [mM]', color='blue')
            ax2.tick_params(axis='y', colors='blue')
            ax2.spines['right'].set_color('blue')

            # Set log scale for x-axis
            ax1.set_yscale('log')
            ax2.set_yscale('log')
            ax1.set_xlabel('Time (h)')
            ax1.grid(visible=True, which="both", linestyle="--", linewidth=0.5)
            ax1.set_title('Substrate concentations over time')

            # Combine legends from both y-axes
            lines = ln1 + ln2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')

            # convert biomass from mol/L to g/L
            c_X1_g = c_X_eD1 * MW_X
            c_X2_g = c_X_eD2 * MW_X
            c_X3_g = c_X_eD3 * MW_X

            axes[1].plot(t,c_X1_g,'-k', label = 'eD1 spec.')
            axes[1].set_xlabel('Time (h)')
            axes[1].set_ylabel('Biomass conc.  (g/L)')
            axes[1].plot(t,c_X2_g,'-g', label = 'eD2 spec.')
            axes[1].plot(t,c_X3_g,'-y', label = 'Generalist')
            axes[1].legend()
            axes[1].grid('True')
            axes[1].set_title('Biomass concentrations over time')

            axes[2].plot(t, f_X1, '-k', label='eD1 specialist')
            axes[2].plot(t, f_X2, '-g', label='eD2 specialist')
            axes[2].plot(t, f_X3, '-y', label='Generalist')
            axes[2].legend()
            axes[2].set_title('Abundance of functional groups over time')
            axes[2].set_xlabel('Time (h)')
            axes[2].set_ylabel('Contribution of functional group to total (x100 %)')
            axes[2].grid('True')
            axes[2].set_ylim((0,1))

            fig.suptitle(f'CSTR behaviour over time at D = {D:.2f} h$^{{-1}}$, f$_{{feed}}$ = {f_feed:.2f} and f$_{{eD1}}$ = {f_eD1:.2f}', fontsize=16)
            fig.tight_layout()

            # second plot
            fig, axs = plt.subplots(1, 3, figsize=(18,6))  # 1 row, 3 columns

            # Subplot 1
            axs[0].plot(t/HRT, c_eD1 * 1000)
            axs[0].grid('True')
            axs[0].set_title('$C_{eD1}$ [mM]')
            axs[0].set_xlabel('Time (HRT)')
            axs[0].set_ylabel('$C_{eD1}$ [mM]')

            # Subplot 2
            axs[1].plot(t/HRT, c_eD2 * 1000)
            axs[1].grid('True')
            axs[1].set_title('$C_{eD2}$ [mM]')
            axs[1].set_xlabel('Time (HRT)')
            axs[1].set_ylabel('$C_{eD2}$ [mM]')

            # Subplot 3
            axs[2].plot(t/HRT, c_X1_g + c_X2_g + c_X3_g)
            axs[2].grid('True')
            axs[2].set_title('Total biomass over time')
            axs[2].set_xlabel('Time (HRT)')
            axs[2].set_ylabel('$C_{X,tot}$ [g/L]')

            fig.suptitle(f'CSTR behaviour over time at D = {D:.2f} h$^{{-1}}$, f$_{{feed}}$ = {f_feed:.2f} and f$_{{eD1}}$ = {f_eD1:.2f}', fontsize=16)

            plt.tight_layout()
            plt.show()

            # Create one figure with two subplots in a row
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # --- Left plot: Total biomass over time ---
            ax1.stackplot(t/HRT, c_X1_g, c_X2_g, c_X3_g, labels=['Spec. eD1', 'Spec. eD2', 'Generalist'])
            ax1.set_xlabel('Time (HRT)')
            ax1.set_ylabel('Biomass concentration [g/L]')
            ax1.set_title(f'Total biomass')
            ax1.legend()
            ax1.grid(True)

            # --- Right plot: Composition over time ---
            ax2.stackplot(t/HRT, f_X1, f_X2, f_X3, labels=['Spec. eD1', 'Spec. eD2', 'Generalist'])
            ax2.set_xlabel('Time (HRT)')
            ax2.set_ylabel('Contribution of functional group to total (x100 %)')
            ax2.set_title(f'Community composition')
            ax2.legend()
            ax2.grid(True)
            fig.suptitle(f'Total biomass and community consumption over time at D = {D:.2f} h$^{{-1}}$, f$_{{eD1,feed}}$ = {f_feed:.2f} and f$_{{eD1,Met,gen}}$ = {f_eD1:.2f}', fontsize=16)
            plt.tight_layout()
            plt.show()

        if show_kin_figures == True:
            for fig in kin_figs:
                display(fig)

    return sol, kin_figs

def survives_species3(c_X3, n_last=500):
    """
    Species 3 survives if:
      1) final biomass > initial biomass
      2) slope of last n_last points >= 0
    """
    # Condition 1: net positive change
    if c_X3[-1] <= c_X3[0]:
        return False
    
    # Extract last N points
    if len(c_X3) < n_last:
        n_last = len(c_X3) // 2  # fallback
    
    y = c_X3[-n_last:]
    x = np.arange(n_last)

    # Fit slope
    slope, _ = np.polyfit(x, y, 1)
    print(slope)

    # Condition 2: slope must not be negative
    epsilon = 1e-10

    if slope < -epsilon:
        return False

    return True
