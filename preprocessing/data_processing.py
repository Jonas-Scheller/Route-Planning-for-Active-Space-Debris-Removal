import pykep as pk
import numpy as np
from tqdm.auto import tqdm
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

import os
import pickle

def filter_TLE_set(PATH_TO_TLE, NEW_FILE_PATH, satellite_name):
    """ filters the tle set for satellite_name """

    debris_ids = []
    line1_dict = {}
    line2_dict = {}

    with open(NEW_FILE_PATH, "w+") as outp:
        with open(PATH_TO_TLE, "r") as inp:
            lines = [r.replace('\n', '') for r in inp.readlines()]

            for i in range(1, int(len(lines)/3)):

                debris_name = lines[3*i]
                line1 = lines[3*i + 1]
                line2 = lines[3*i + 2]

                if debris_name == satellite_name:
                    outp.write(debris_name + '\n')
                    outp.write(line1 + '\n')
                    outp.write(line2 + '\n')

def convert_to_epoch(day, month, year):
    """ returns epoch for day month and year """

    day = str(day)
    month = str(month)
    year = str(year)

    if len(day) < 2:
        day = "0" + day

    if len(month) < 2:
        month = "0" + month

    return pk.epoch_from_string(year + '-' + month + '-' + day + ' 23:59:54.003')

def distance_static_TSP(o1, o2):
    """ compute three impulse approximation for two planets """
    return pk.phasing.three_impulses_approx(o1, o2)

def distance_dynamic_TSP_three_impulses(o1, o2, t0):
    """ compute three impulse approximation at epoch t0 for two planets """
    return pk.phasing.three_impulses_approx(o1, o2, ep1 = pk.epoch(t0), ep2=pk.epoch(t0+1))

def distance_dynamic_TSP_lambert(o1, o2, t0, T):
    """ compute lambert problem at epoch t0 arriving at t0 + T """

    r1,v1 = o1.eph(t0)
    r2,v2 = o2.eph(t0 + T)
    l = pk.lambert_problem(r1,r2,T)

    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1_l = np.asarray(l.get_v1()[0])
    v2_l = np.asarray(l.get_v2()[0])

    c = np.linalg.norm(v1 - v1_l, ord = 1) + np.linalg.norm(v2 - v2_l, ord = 1)

    return c

def add_leading_zero_static(A, P):
    """ append zero row and column to A and P of static TSP instance """
    A_new = np.zeros((A.shape[0] + 1, A.shape[1] + 1))
    P_new = np.zeros(P.shape[0] + 1)

    A_new[1:,1:] = A
    P_new[1:] = P

    return A_new, P_new

def add_leading_zero_dynamic(A, P, params):
    """ append zero row and column to A, P and params of static TSP instance """

    A_new = np.zeros((A.shape[0] + 1, A.shape[1] + 1, A.shape[2]))
    P_new = np.zeros(P.shape[0] + 1)
    params_new = np.zeros((params.shape[0] + 1, params.shape[1] + 1, params.shape[2]))

    A_new[1:, 1:, :] = A
    P_new[1:] = P
    params_new[1:, 1:, :] = params

    return A_new, P_new, params_new

def get_radarArea(planets, satcat):
    """ returns radarA area of planets as in satcat catalogue """
    radarArea = {}
    for p in planets:
        id = p.name.strip()
        radarArea[id] = satcat[id].radarA
    return radarArea

def filter_non_decayed_static(planets):
    """ filters out planets for which no three impulse approximation is
        possible, i.e., decayed satellites """
    filtered_planets = []

    print("Preprocessing indices for three impulse approximation")
    for p1 in tqdm(planets):
        success = False

        for p2 in planets:
            if p1.name!=p2.name:
                try:
                    pk.phasing.three_impulses_approx(p1, p2)
                    success |= True
                except:
                    continue

        if success:
            filtered_planets.append(p1)

    return filtered_planets

def filter_non_decayed_dynamic(planets_inp, epochs):
    """ filters out planets for which no three impulse approximation is
        possible, i.e., decayed satellites """
    planets = planets_inp[:]

    blacklist = set([])
    nPlanets = len(list(planets))
    error = np.zeros((nPlanets, nPlanets))

    print("Preprocessing indices for dynamic three impulse approximation")
    for i, p1 in enumerate(tqdm(planets)):

        for t in epochs:
            e_start = pk.epoch(t, "mjd2000")
            e_arrive = pk.epoch(t+1, "mjd2000")
            success = False
            for j, p2 in enumerate(planets):
                if p1.name!=p2.name:
                    try:
                        pk.phasing.three_impulses_approx(p1, p2, ep1 = e_start, ep2 = e_arrive)
                    except:
                        error[i,j] = 1
                        continue

    for i, p in enumerate(planets):
        sum_col = sum(error[:,i])
        sum_row = sum(error[i,:])

        if sum(error[:,i]) > 0.1 * nPlanets and sum(error[i,:]) > 0.1 * nPlanets:
            blacklist.add(p.name.strip())

    filtered_planets = [p for p in planets_inp if p.name.strip() not in blacklist]

    return filtered_planets

def compute_A_static_three_impulse(planets):
    """ computes static weight matrix for planets with three impulse approximation """
    n = len(planets)
    A = np.zeros((n,n))

    print("Computing cost approximations for static TSP")
    for i, p1 in enumerate(tqdm(planets)):
        for j, p2 in enumerate(planets):
            if i!=j:
                A[i,j] = pk.phasing.three_impulses_approx(p1, p2)

    return A

def compute_A_dynamic_three_impulse(epochs, planets):
    """ computes dynamic weight matrix for planets with three impulse approximation """
    n = len(planets)
    max_time = len(epochs)
    A = np.zeros((n,n,max_time))

    print("Computing cost approximations for dynamic TSP")

    for i, p1 in enumerate(tqdm(planets)):
        for t,e in enumerate(epochs):
            e_start = pk.epoch(e, "mjd2000")
            e_arrive = pk.epoch(e+1, "mjd2000")
            for j, p2 in enumerate(planets):
                if i!=j:
                    A[i,j,t] = pk.phasing.three_impulses_approx(p1, p2, ep1 = e_start, ep2 = e_arrive)

    return A

def compute_P_radar(debris_ids, satcat):
    """ computes preference matrix by radar value of satcat catalogue """
    P = np.zeros(len(debris_ids))

    for i,id in enumerate(debris_ids):
        try:
            P[i] = satcat[id].radarA
        except:
            P[i] = 0
    return P

def compute_epochs(start, steps, step_size):
    """ returns list of epochs, starting at start (juliandate from 2000)
        with steps steps and step_size days inbetween the steps """
    epochs = []

    for s in range(steps):
        curr = start + s*step_size
        epochs.append(curr)

    return epochs

def compute_regression_for_matrix(A):
    """ computes slopes and intercepts for regression on matrix A """
    n = A.shape[0]
    t_max = A.shape[2]

    l_params = np.zeros((n,n,2))

    print("Computing regression parameters")
    for i in tqdm(range(n)):
        for j in range(n):
            if i != j:
                x = np.arange(t_max).reshape(-1,1)
                y = A[i,j,:]
                reg = LinearRegression().fit(x, y)
                l_params[i,j,0] = reg.predict([[0]])
                l_params[i,j,1] = reg.coef_[0]

    return l_params

def compute_static_TSP_data(SATCAT_PATH, TLE_PATH):
    """ computes static TSP instance for satellites in TLE_PATH """
    planets = pk.util.read_tle(TLE_PATH, with_name=True)
    satcat = pk.util.read_satcat(SATCAT_PATH)

    non_decayed = filter_non_decayed_static(planets)
    debris_ids = [p.name.strip() for p in non_decayed]

    A = compute_A_static_three_impulse(non_decayed)
    P = compute_P_radar(debris_ids, satcat)

    res = {}
    res['A'] = A
    res['P'] = P
    res['ids'] = debris_ids
    res['line1'] = [p.line1 for p in non_decayed]
    res['line2'] = [p.line2 for p in non_decayed]

    return res

def compute_dynamic_TSP_data(SATCAT_PATH, TLE_PATH, epochs):
    """ computes dynamic TSP instance for satellites in TLE_PATH """
    planets = pk.util.read_tle(TLE_PATH, with_name=True)
    satcat = pk.util.read_satcat(SATCAT_PATH)

    # filter out all planets, where sgp4 propagation fails
    non_decayed = filter_non_decayed_dynamic(planets, epochs)
    debris_ids = [p.name.strip() for p in non_decayed]

    A = compute_A_dynamic_three_impulse(epochs, non_decayed)
    P = compute_P_radar(debris_ids, satcat)

    res = {}
    res['A'] = A
    res['P'] = P
    res['epochs'] = epochs
    res['ids'] = debris_ids

    return res

def compute_static_TSP_instance(SATCAT_PATH, TLE_PATH, nMaxNodes = -1):

    planets = pk.util.read_tle(TLE_PATH, with_name=True)
    satcat = pk.util.read_satcat(SATCAT_PATH)

    non_decayed = filter_non_decayed_static(planets)
    debris_ids = [p.name.strip() for p in non_decayed]

    A = compute_A_static_three_impulse(non_decayed)
    P = compute_P_radar(debris_ids, satcat)

    n = nMaxNodes
    if nMaxNodes == -1:
        n = A.shape[0]

    sorted_ind = np.argsort(P).reshape(-1)[::-1]
    A = A[sorted_ind,:][:,sorted_ind]
    P = P[sorted_ind]

    A,P = add_leading_zero_static(A,P)

    return A, P

def compute_dynamic_TSP_instance(SATCAT_PATH, TLE_PATH, startDay, nEpochs, step_size = 7, nMaxNodes = -1, withRegression = False):

    epochs = compute_epochs(startDay, nEpochs, step_size)
    dynamic_data = compute_dynamic_TSP_data(SATCAT_PATH, TLE_PATH, epochs)

    A_raw, P_raw = dynamic_data['A'], dynamic_data['P']

    par_raw = np.zeros((A_raw.shape[0], A_raw.shape[1], 2))
    if withRegression:
        par_raw = compute_regression_for_matrix(A_raw)

    A, P, par = add_leading_zero_dynamic(A_raw, P_raw, par_raw)

    n = nMaxNodes
    if nMaxNodes == -1:
        n = A.shape[0]

    # sort by priority
    sorted_ind = np.argsort(P).reshape(-1)[::-1]
    A = A[sorted_ind,:,:][:,sorted_ind,:]
    P = P[sorted_ind]
    par = par[sorted_ind,:,:][:,sorted_ind,:]

    return A[:n,:n,:], P[:n], par[:n,:n,:]


def pickle_data(data, filename):
    """ pickle data under filename """
    if not os.path.exists("processed"):
        os.makedirs("processed")

    with open("processed/" + filename + ".p","wb") as f:
        pickle.dump(data,f)

def unpickle_data(filename):
    """ unpickle data under filename """
    with open("processed/" + filename + ".p","rb") as f:
        return pickle.load(f)
