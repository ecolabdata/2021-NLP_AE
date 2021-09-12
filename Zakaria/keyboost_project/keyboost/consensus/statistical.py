import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
from warnings import filterwarnings
filterwarnings('ignore')
from keyboost.consensus.utils import deduplication
from keyboost.consensus.ranking import rank_consensus


# Allows to set an optimal scale for the histogram in terms of bins
def freedman_diaconis_rule(data):
    a = np.array(data)
    if len(data) <2:
        return 1
    h = 2*(np.percentile(data,75)-np.percentile(data,25))*len(data)**(-1/3)

    if h==0:
        return len(data)
    else:
        return int(np.ceil((max(data)-min(data))/h))



# Estimation of the parameters of an a priori selected statistical model
def maximum_likelyhood_estimation(data,
                                  distribution):

    params = distribution.fit(data)

    arg = params[:-2] # distribution parameter vector
    loc = params[-2] # location parameter
    scale = params[-1] # scale parameter

    return arg, loc, scale


# Selects the best discriminative statistical model in terms of the SSE metric

def DSM_selection(data,
                 distributions,
                 bins):

    y, x = np.histogram(a=data,
                       bins=bins,
                       density=True)
    x = (x + np.roll(x,-1))[:-1] / 2.0

    # By default, we suppose that the data fits a gaussian model

    best_distribution = st.norm
    best_arg,best_loc,best_scale = (None, 0.0, 1.0)
    best_sse = np.inf

    for distribution in distributions:

        arg, loc, scale = maximum_likelyhood_estimation(data,distribution)

        pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
        sse = np.sum(np.power(y-pdf,2.0))

        if best_sse > sse:
            best_distribution = distribution
            best_arg = arg
            best_loc = loc
            best_scale = scale
            best_sse = sse


    return best_distribution, best_arg, best_loc, best_scale


# packs isolated parameters of a statistical model into a single list

def pack_params(arg,loc,scale):
    params = list()
    a = [arg,loc,scale]
    for x in a:
        if type(x)!=tuple:
            params.append(x)
        else:
            for t in x:
                params.append(t)
    return tuple(params)

# performs Kolmogorov Smirnov Test

def kolmogorov_smirnov_test(distribution,
                           data,
                           arg,
                           loc,
                           scale,alpha):

    args = pack_params(arg,loc,scale)

    _, p_value = st.kstest(rvs=data,
                          cdf=distribution.name,
                          args=args,
                          alternative='two-sided')


    return True if p_value >= alpha else False # Pass the test if we do not reject H0 : the score distribution is discriminative


# Checks that the MLE parmaters lies within the boundaries of what we set as discriminative parameters
# for each statistical model

def DSM_params_check(distribution,
                     arg):

    if distribution == st.powerlaw:

        return True if 0<arg[0]<1 else False

    elif distribution == st.gamma:

        return True if 0<arg[0]<15 else False

    elif distribution == st.pareto:

        return True

    elif distribution == st.powerlognorm:

        return True if 0<arg[1]<1 else False

# Consolidation + Global Rank Extraction
def extract_rank(scores):

    key_rank = pd.concat(scores,axis=0).sort_values(by='Score',ascending=False)
    key_rank.index = range(len(key_rank))
    key_rank['Score'] = key_rank['Score']-key_rank['Score'].min()

    return  key_rank


# deduplication and final output

def statistical_consensus(key_extractions,
                          n_top,
                          transformers_model='distilbert-base-nli-mean-tokens',
                          alpha_ks=0.01):

    discriminative_statistical_models = [st.powerlaw,st.gamma,st.pareto,st.powerlognorm]
    discriminative_sanity = list()

    #print('*** statistical discriminativeness check **')
    i=1
    for key_extraction in key_extractions:
        #print('* keyword extraction model {} *'.format(i))
        i+=1

        optimal_bins = freedman_diaconis_rule(data=key_extraction['Score'])
        best_distribution, best_arg, best_loc, best_scale = DSM_selection(data=key_extraction['Score'],
                                                                          distributions=discriminative_statistical_models,
                                                                          bins=optimal_bins)
        DSM_compatible_params = DSM_params_check(best_distribution,best_arg)
        if DSM_compatible_params :
            #print('DSM Params OK')
            #do ks Test
            is_ks_discriminative = kolmogorov_smirnov_test(distribution=best_distribution,
                   data=key_extraction['Score'],
                   arg=best_arg,
                   loc=best_loc,
                   scale=best_scale,
                   alpha=alpha_ks)

            if is_ks_discriminative:
                #print('KS OK')
                #print('Proper DSM behind data')
                discriminative_sanity.append(True)
            else:
                #print('No DSM behind data')
                discriminative_sanity.append(False)
        else:
            #print('No DSM behind data')
            discriminative_sanity.append(False)


    #print(discriminative_sanity)

    if np.array(discriminative_sanity).all():
            #print('Consolidation + Rank Extraction')
            key_rank = extract_rank(key_extractions)
            key_rank = key_rank[:len(key_extractions)*n_top]
            #print('Deduplication')
            keywords = deduplication(key_rank=key_rank,
                  n_top=n_top,
                  transformers_model='distilbert-base-nli-mean-tokens')
            result = pd.DataFrame(keywords,columns=['Keyword','Score'])
            #print('Done !')
            return result

    else:
        #print('Fallback to Rank Based Consensus')
        return rank_consensus(key_extractions,n_top)
