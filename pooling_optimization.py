from random import random
from math import log
import csv

def generate_randomly_sick(p, m):
    """ generates random array of length m of 1s and 0s where 1 has
        probability p
    """
    rand_arr = []
    for i in range(m):        
        rand_num = random()
        is_sick = int((rand_num < p))
        rand_arr.append(is_sick)
    return rand_arr

def prob_of_sick_among_m(p, m):
    """ returns the probability that there exists at least one sick person
        among m independent people, each sick with probability p.
    """
    return (1 - (1-p)**m)

def prob_of_sick_among_m_given_n_are_sick(p, m, n):
    """ For n>=m, given n independent people in a row, each sick with
        probability p, conditioned on the existence of at least one sick
        person among the n people, returns the probability that there
        exists at least one sick person among the left-most m people.
    """
    return prob_of_sick_among_m(p,m) / prob_of_sick_among_m(p,n)

def bin_entropy(p):
    """ calculate binary entropy of a random varible that takes 2 values,
        one of them with probability p
    """
    q = 1 - p
    if p==0 or p==1:
        return 0
    return -p*log(p, 2) -q*log(q,2)

def calc_theoretical_efficiency(p_sick):
    """ Calculates the theoretical highest testing efficiency using entropy
        bounds. Efficiency is measured as the number of independent
        people, each sick with probability p_sick, divided by the expected
        number of tests needed to decide the status of some person, so that
        an efficiency of 
    """
    entropy = bin_entropy(p_sick)
    return 1 / entropy
    

def calc_pool_group_efficiency(p_sick, pool_size, d=1):
    """ Calculates the efficiency of a two-stage pooling algorithm placing
        people in a d-dimensional cube of side length pool_size, containing
        pool_size^d people. d=1 corresponds to the simple Dorfman algorithm.
        The algorithm: each generalized row is tested (for example, if d=2 then
        each row and each column of the matrix is tested), and then test
        individually each person for which all the generalized rows test
        positive.
        Efficiency is defined as the number of people in the d-dimensional cube
        divided by the expected number of tests, when people are assumed to be
        independent and sick with probability p_sick.
    """
    # FIRST STAGE: test each row. Each person is in d generalized rows
    # and each generalized row contains pool_size people.
    num_tests_first_stage = d * pool_size**(d-1)
    
    q_sick = 1- p_sick
    expected_num_tests_second_stage = (pool_size**d) * (p_sick + q_sick * ((1 - q_sick**(pool_size-1))**d))
    expected_num_tests = num_tests_first_stage + expected_num_tests_second_stage
    return pool_size**d / expected_num_tests


def optimize_two_stage(p_sick, MAX_DIM = 6, MAX_POOL_SIZE = 1000):
    """ chooses the best pool size for each dimension, for a two-stage
        pooling algorithm on independent people with p_sick probability of
        being sick, up to pool size MAX_POOL_SIZE and up to cube dimension
        MAX_DIM. (dimension 1 is the simplest Dorfman algorithm)
        Returns two arrays (best_effs, best_pool_sizes) of size MAX_DIM each,
        where best_effs[d] and best_pool_sizes[d] are the optimal efficiency
        and optimal pool size for a testing scheme of a d-dimensional cube.
    """
    
    best_effs = []
    best_pool_sizes = []
    for dimension in range(1, MAX_DIM+1):
        cur_dim_best_eff = -1
        cur_dim_best_pool_size = -1
        for pool_size in range(2, MAX_POOL_SIZE+1):
            cur_eff = calc_pool_group_efficiency(p_sick, pool_size, dimension)
            if cur_eff > cur_dim_best_eff:
                cur_dim_best_eff = cur_eff
                cur_dim_best_pool_size = pool_size
        best_effs.append(cur_dim_best_eff)
        best_pool_sizes.append(cur_dim_best_pool_size)
    return best_effs, best_pool_sizes
    
        
def write_optimization_to_excel(filepath, min_p, max_p, num_p_values,
                                MAX_DIM = 6, MAX_POOL_SIZE = 1000):
    """ Writes a csv file to location filepath containig the optimization
        results from the function optimize_two_stage, when it is run on
        a range of num_p_values apriori probabilities, from  min_p to max_p.
    """
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')

        # write first line with calculation parameters
        first_line = ["""Two stage optimization for: min_p=%2.2f, max_p=%2.2f,
                        num_p_values=%d, MAX_DIM=%d, MAX_POOL_SIZE=%d"""%(min_p,max_p,num_p_values,MAX_DIM,MAX_POOL_SIZE)]
        writer.writerow(first_line)

        # write table headers
        headers = ['p sick', 'max theoretic eff', 'best dim', 'best eff','best pool size']
        for dim in range(1, MAX_DIM+1):
            headers.append('d=%d best eff'%dim)
            headers.append('d=%d best pool size'%dim)
        
        writer.writerow(headers)

        # write optimizations
        cur_p = min_p
        p_diff = (max_p - min_p) / float(num_p_values)
        while(cur_p <= max_p):
            cur_max_theoretic_eff = calc_theoretical_efficiency(cur_p)
            cur_row = ['%3.3f'%cur_p, '%3.3f'%cur_max_theoretic_eff]
            cur_best_effs, cur_best_pool_sizes = optimize_two_stage(cur_p, MAX_DIM, MAX_POOL_SIZE)
            abs_best_eff = max(cur_best_effs)
            abs_best_ind = cur_best_effs.index(abs_best_eff)
            abs_best_dim = abs_best_ind + 1
            abs_best_pool_size = cur_best_pool_sizes[abs_best_ind]
            cur_row += [abs_best_dim, '%2.3f'%abs_best_eff, abs_best_pool_size]
            for dim in range(0, MAX_DIM):
                cur_row.append('%2.3f'%cur_best_effs[dim])
                cur_row.append(cur_best_pool_sizes[dim])
            writer.writerow(cur_row)
            cur_p += p_diff


def simulate_dorfman(num_times, p_sick, m):
    """ simulates num_times the simple Dorfman pooling algorithm:
        generates num_times a group of m independent people randomly sick with
        probability p_sick, and if group is positive tests it individually.
    """
    tot_tests = 0
    tot_num_sick = 0
    for i in range(num_times):
        # create group of randomly sick people
        cur_num_sick = 0
        cur_group = generate_randomly_sick(p_sick,m)
        tot_num_sick += sum(cur_group)

        # run test on group
        group_test = (sum(cur_group) > 0)
        if group_test:
            tot_tests += m + 1
        else:
            tot_tests += 1

    avg_sick = tot_num_sick / (float(num_times)*m)
    avg_tests = tot_tests / float(num_times)
    print("avg_sick:", avg_sick)
    print("avg_tests:", avg_tests)
    print("avg_test_efficiency:", float(m) / avg_tests)

def calc_Dorfman_efficiency(p_sick, m):
    """ caculates the efficency of simple Dorfman pooling with pool size m for
        an independent and identically distributed samples where each sample
        has probability p_sick of testing positive.
    """
    return calc_pool_group_efficiency(p_sick, m, 1)

def calc_matrix_efficiency(p_sick, m, retest_deduceable = True):
    """ caculates the efficency of matrix pooling with pool size m for
        an independent and identically distributed samples where each sample
        has probability p_sick of testing positive.
        If retest_deduceable is False, then samples matrices with at most
        one positive row or at most one positive column are not retested.
    """
    if retest_deduceable:
        return calc_pool_group_efficiency(p_sick, m, 2)
    else:
        q = 1-p_sick
        inv_eff = 2./m + (1-q**m)*(1-q**(m**2-m)) - \
               q**m*(1-q**(m-1))*(1-q**((m-1)**2)+p_sick*q**(m**2-2*m))
        return 1. / inv_eff

def optimize_matrix_without_retesting_deduceable(p_sick, MAX_POOL_SIZE = 1000):
    """ Chooses the best pool size for each dimension, for a two-stage
        matrix pooling algorithm, where matrices with at most one positive
        row or at most one positive column are not retested.
        People are assumed to be independent people with p_sick probability
        of being sick, up to pool size MAX_POOL_SIZE.
        Returns the best efficiency and the corresponding pool size.
    """
    best_eff = -1
    best_pool_size = -1
    for pool_size in range(2, MAX_POOL_SIZE+1):
        cur_eff = calc_matrix_efficiency(p_sick, pool_size, False)
        if cur_eff > best_eff:
            best_eff = cur_eff
            best_pool_size = pool_size
    return best_eff, best_pool_size


def simulate_matrix(num_times, p_sick, m, retest_deduceable = True):
    """ simulates num_times the a two-stage matrix pooling algorithm:
        generates num_times a group of m**2 independent people randomly sick
        with probability p_sick, and places them in an m-by-m matrix.
        Tests each row and each column, and then tests individuals
        such that both their row and column tested positive.
        If retest_deducable is False, and there is at most one positive
        row or at most one positive column, no retesting is done as
        the positive people in the matrix can be deduced.
    """
    tot_tests = 0
    tot_num_sick = 0
    for i in range(num_times):
        # create matrix of randomly sick people
        mat = []
        for i in range(m):
            cur_row = generate_randomly_sick(p_sick,m)
            mat.append(cur_row)
            tot_num_sick += sum(cur_row)

        # run tests on rows
        row_tests = [sum(x)>0 for x in mat]
        col_tests = [sum([x[j] for x in mat])>0 for j in range(m)]

        # calculate num of suspected positives
        sus_pos = sum(row_tests) * sum(col_tests)
        if retest_deduceable == False and \
           (sum(row_tests)<=1 or sum(col_tests)<=1):
            sus_pos = 0
        cur_tests = 2*m + sus_pos
        tot_tests += cur_tests

    avg_sick = tot_num_sick / (float(num_times) * (m**2))
    avg_tests = tot_tests / float(num_times)
    print("avg_sick:", avg_sick)
    print("avg_tests per matrix:", avg_tests)
    print("avg_test_efficiency:", float(m**2) / avg_tests)
