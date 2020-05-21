# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:27:44 2018

@author: dizhang
"""


# this is one of the script of a project regarding inventory replenishment.
# The goal is to find the best strategy to buy a bunch of items from a few vendors,
# in order to minimize item costs and opportunity cost (out of stock loss)
# with each vendor having their own discount policy (free shipping over $XXX etc).
# Under certain circumstances, we'll buy few surplus items in order to reach the threshold for free shipping.
# This script is about how we'll use Generic Algorithm to find the best strategy.



import numpy as np
import pandas as pd
import traceback
import random
import re
#from deap import creator as creator, base, tools, algorithms
#from deap import base, tools, algorithms
from deap import creator as creator_mod, base, tools, algorithms

from classDef_Vendor import Vendor
from strategies import buy_random, all_permutation_iter, random_permutation_iter, get_until_reach_threshold
from output_analysis import part_and_opportunity_cost, shipping_cost_by_vendor

import logging
logger = logging.getLogger('runStrategy')
#from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool, cpu_count









def run_all_strategies(strategies, cuc_vendor_avail, cuc_premium_demand, vendor,
                       surplus_avail = None, surplus_budget = 0,
                       random_repeat = 5, vd_perm_max = 6000, print_traceback = False, multiprocess = False, creator = None):
    '''run all given strategies for given cuc premium demand, and vendor availability
    
    A typical buying process is to go over freight-possible vendors in a certain order, 
    and allocate parts when it's their turn with a certain strategy (e.g. from the cheapest) until pre-paid freight is made. Calculate cost function afterwards.
    We repeat the process for multiple times and choose the best permutation with lowest Cost.
    
    Number of freight-possible vendor is calculated first. If >= 7, then we use Genetic Algorithm to find the best permutation. If < 7, then exhaust search them.
    If using buy_random strategy, we repeat it for random_repeat times, each time only pick no more than 6000 random vendor permutations.
    
    
    Args:    
        strategies (list of functions): a list of strategy function names
        cuc_vendor_avail (pandas.DataFrame)
        cuc_premium_demand (pandas.DataFrame)
        vendor (pandas.DataFrame)
        surplus_avail (pandas.DataFrame)
        surplus_budget (float)
        random_repeat: if using 'buy_random' strategy, this is the number of repetition, meaning we'll run buy_random strategy for random_repeat times.
        vd_perm_max (int): if using 'buy_random' strategy, this is the number of random vendor permutation generated in each run
        print_traceback: whether printing traceback if there's an error
        multiprocess: whether to multiprocess while running genetic algorithm
        creator: creator object associated with genetic algorithm (if multiprocessing, it has to be passed from main script)
    '''
    
    # initiate output placeholders
    outputs = []
    costs = []
    strategy_list = []
    vd_perm_list = []
    bought_surplus_list = []
    
    # whether it's possible for a vendor to make freight
    # vendor | freight_type | freight_value | possibility | highest_poss_price | highest_poss_qty
    vendor_freight_poss = vendor_freight_possibility(cuc_vendor_avail, vendor, 
                                                     surplus_avail, surplus_budget)
    
    # cuc | willingness | vendor | price
    # of all the vendors that it's possible that we make freight
    cuc_vendor_avail_freight = \
    cuc_vendor_avail.merge(vendor_freight_poss.loc[vendor_freight_poss.possibility == True, \
                                                  ['vendor', 'possibility']], \
                           on = 'vendor')
    
    #all freight-possible vendors
    v = np.unique(cuc_vendor_avail_freight.vendor)
    logger.info('freight_vendors: {0}'.format(v))
    
    # no freight vendor
    if len(v) == 0:
        try:
            output, cost, checked_status, bought_surplus = run_strategy_vdperm('', [], cuc_vendor_avail, cuc_vendor_avail_freight, 
                                                               cuc_premium_demand, vendor, 
                                                               surplus_avail, surplus_budget)
            outputs.append(output)
            costs.append(cost)
            strategy_list.append('no_freight_vendor')
            vd_perm_list.append(None)
            bought_surplus_list.append(bought_surplus)
        except Exception as e:
            print(error_msg_run_strategy(e, 'no_freight_vendor', 'NA', print_traceback))
            
        return outputs, costs, strategy_list, vd_perm_list, bought_surplus_list
    

    # freight vendor exists (and not too many)
    for strategy in strategies:
        
        if strategy != buy_random:
            logger.info('strategy:' + strategy.__name__)
            checked_status = {}                
            count = 0
            
            # set up iterators
            if len(v) > 6:
                logger.info('Freight vendor count: {0}. Use Generic Algorithm to find optimal vendor permutation.'.format(len(v)))
                try:
                    output, cost, bought_surplus, vd_perm = run_strategy_ga(strategy, v, cuc_vendor_avail, 
                                                                   cuc_vendor_avail_freight, cuc_premium_demand, vendor, 
                                                                   surplus_avail, surplus_budget, multiprocess = multiprocess, creator = creator)
                    # append results
                    outputs.append(output); costs.append(cost); strategy_list.append(strategy.__name__); 
                    vd_perm_list.append(vd_perm); bought_surplus_list.append(bought_surplus)
                except Exception as e:
                    logger.warning(error_msg_run_strategy(e, 'Genetic Algo', 'NA', print_traceback))
                    
            elif len(v) > 0:
                vdperm_iterator = all_permutation_iter(v)
                logger.info('Freight vendor count: {0}. Use Brutal Force to find optimal vendor permutation.'.format(len(v)))
                # iterating
                for vd_perm in vdperm_iterator:
                    # check if current permutation can generate a different result from previous    
                    if skip_vd_perm(vd_perm, checked_status):
                        continue
                    
                    try:
                        output, cost, checked_status, bought_surplus = run_strategy_vdperm(strategy, vd_perm, cuc_vendor_avail, 
                                                                           cuc_vendor_avail_freight, cuc_premium_demand, vendor, 
                                                                           surplus_avail, surplus_budget)
                        # append results
                        outputs.append(output); costs.append(cost); strategy_list.append(strategy.__name__); 
                        vd_perm_list.append(vd_perm); bought_surplus_list.append(bought_surplus)
                    except Exception as e:
                        logger.warning(error_msg_run_strategy(e, strategy.__name__, vd_perm, print_traceback))
        
                    
    # if buy_random included in strategies, do it for random_repeat times
    if buy_random in strategies:
        for k in range(random_repeat):
            logger.info('strategy: buy_random Repetition:' + str(k+1))
            checked_status = {}                
            count = 0
            
            # set up iterators
            if len(v) > 6:
                logger.info('Freight vendor count: {0}. Only run for a random subset of vendor permutations.'.format(len(v)))
                vdperm_iterator = random_permutation_iter(v, vd_perm_max)
            elif len(v) > 0:
                vdperm_iterator = all_permutation_iter(v)
            
            # iterating
            for vd_perm in vdperm_iterator:
                count += 1
                if count % 1000 == 0: print(str(int(count)) + ' permutations processed.')
                
                # check if current permutation can generate a different result from previous                        
                if skip_vd_perm(vd_perm, checked_status):
                    continue
                
                try:
                    output, cost, checked_status, bought_surplus = run_strategy_vdperm(buy_random, vd_perm, cuc_vendor_avail, 
                                                                       cuc_vendor_avail_freight, cuc_premium_demand, vendor,
                                                                       surplus_avail, surplus_budget)
                    # append results
                    outputs.append(output); costs.append(cost); strategy_list.append(strategy.__name__); 
                    vd_perm_list.append(vd_perm); bought_surplus_list.append(bought_surplus)
                except Exception as e:
                    logger.warning(error_msg_run_strategy(e, 'buy_random', vd_perm, print_traceback))

    logger.info('Run strategies completed.')
    return outputs, costs, strategy_list, vd_perm_list, bought_surplus_list
    









def vendor_freight_possibility(cuc_vendor_avail, vendor, surplus_avail = None, surplus_budget = 0):
    '''
    whether it's possible for a vendor to make freight (with CUCs in need and possible surplus items)
    
    Args:
        cuc_vendor_avail (pandas.DataFrame)
        vendor (pandas.DataFrame)
        surplus_avail (pandas.DataFrame)
        surplus_budget (pandas.DataFrame)
        
    Returns: 
        vendor | freight_type | freight_value | possibility | highest_poss_price | highest_poss_qty
    '''
    # qty = 1 for single pieces, qty = N for batch indicating number of batch, but price = extended price
    
    # total quantity and dollar amount possible for each vendor
    vendor_freight_poss = cuc_vendor_avail[['vendor', 'price', 'qty']].groupby('vendor').sum().reset_index()\
                                  .merge(vendor, on = 'vendor')
    
    # total surplus quantity and surplus dollar amount possible for each vendor
    if surplus_avail is not None and len(surplus_avail) > 0:
        surplus_values = surplus_avail[['vendor', 'price', 'qty']]\
                                      .groupby('vendor').sum().reset_index()\
                                      .merge(vendor, on = 'vendor')\
                                      .rename(columns = {'qty': 'qty_s', 'price': 'price_s'})
    else:
        surplus_values = pd.DataFrame(columns = ['vendor', 'price_s', 'qty_s'])
    
    # join the ttl qty/amount for needed items and for surplus items
    vendor_freight_poss = vendor_freight_poss.merge(surplus_values, how = 'left')       
    vendor_freight_poss.price_s[pd.isnull(vendor_freight_poss.price_s)] = 0                           
    vendor_freight_poss.qty_s[pd.isnull(vendor_freight_poss.qty_s)] = 0
    
    # freight possibility: (price here means extended price)
    # if amount type freight: price > freight $amount - surplus budget
    # if qty type freight: generic qty + surplus qty > freight qty                                                      
    vendor_freight_poss['possibility'] = [row.price >= row.freight_value - surplus_budget and row.price + row.price_s > row.freight_value
                                          if row.freight_type == 'amount' 
                                          else row.qty + row.qty_s >= row.freight_value \
                                          for row in vendor_freight_poss.itertuples()]
    
    # join original vendor dataframe to get all vendors
    vendor_freight_poss = vendor.merge(vendor_freight_poss[['vendor', 'possibility', 'price', 'qty', 'price_s', 'qty_s']],\
                                       on = 'vendor', how = 'left')
    vendor_freight_poss = vendor_freight_poss.rename(columns = {'price': 'highest_poss_price', \
                                                                'qty': 'highest_poss_qty',\
                                                                'price_s': 'highest_surplus_price',\
                                                                'qty_s': 'highest_surplus_qty'})    
    return vendor_freight_poss









def skip_vd_perm(vd_perm, checked_status):
    '''
    check if current permutation can generate a different result from previous  
    e.g. Consider premutations ACDEFB and ACDEBF
    if we go through ACDEFB first and realized that after part allocation for ACDE, F and B are both not freight possible, then we can skip ACDEBF.
    

    Args:
        vd_perm: current vd permutation
        checked_status (dictionary): {'checked_vendor': vendor order list in last run, 'freight_possible': list of remaining vendors that are still possible to make pre-paid freight}
    '''
    if checked_status:
        skip = True
        # if current permutation and last permutation run has same order for freight vendors
        for i in range(len(checked_status['checked_vendor'])):
            skip = skip and checked_status['checked_vendor'][i] == vd_perm[i]
        # and if the next vendor in current permutation is not in the group of vendors that's possible to make freight with last run's freight vendors selected
        skip = skip and vd_perm[i+1] not in checked_status['freight_possible']
        return skip





def error_msg_run_strategy(e, strategy_name, vd_perm, print_traceback = False):
    '''
    error message
    
    Args:
        e (exception object): exception thrown.
        strategy_name (str)
        vd_perm (list of string)
        print_traceback (bool)
    
    '''
    error_msg = \
    '''
    run strategy failed for:
        strategy: {0}
        vendor permutation: {1}
        Error message: 
            {2}
    '''.format(strategy_name, vd_perm, e)
    
    if print_traceback:
        error_msg = error_msg + \
            '''
            {0}
            '''.format(traceback.format_exc())
    return error_msg
    

#creator = None
#def set_creator(cr):
#    global creator
#    creator = cr


def run_strategy_ga(strategy, v, cuc_vendor_avail, 
                    cuc_vendor_avail_freight, cuc_premium_demand, vendor, 
                    surplus_avail, surplus_budget, multiprocess = False, creator = None):
    """ run genetic algorithm to find best vendor permutation with a given strategy
    
    refer to package DEAP https://deap.readthedocs.io/en/master/index.html for more information on GA.
    
    Args:
        strategey (function object)
        v (list of str): list of freight-possible vendors
        cuc_vendor_avail (pandas.DataFrame)
        cuc_vendor_avail_freight (pandas.DataFrame)
        cuc_premium_demand (pandas.DataFrame)
        vendor (pandas.DataFrame)
        surplus_avail (pandas.DataFrame)
        surplus_budget (float)
        multiprocess (bool): whether we want to multiprocess or not
        creator (DEAP.creator object): passed in from main script. Used only when multiprocess is True.
        
    
    """
    IND_SIZE=len(np.unique(cuc_vendor_avail_freight.vendor))   #number of freight vendors
    POP_SIZE=200
    NGEN=50
    logger.info('Gneric Algorithm settings: Population Size: {0}, Max Generation: {1}'.format(POP_SIZE, NGEN))

    # registering tools
    if not multiprocess:
        creator = creator_mod
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
    #logger.info('creator:{0}'.format(creator))
    
    toolbox = base.Toolbox()

    if multiprocess:
        logger.info('Use Multiprocessor')
        pool = Pool(cpu_count()-1)
        toolbox.register('map', pool.map)
    
    toolbox.register("indices", random.sample, range(IND_SIZE), IND_SIZE)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", cost_wrapper, v = v,
                     strategy = strategy, cuc_vendor_avail = cuc_vendor_avail, 
                     cuc_vendor_avail_freight = cuc_vendor_avail_freight, 
                     cuc_premium_demand =cuc_premium_demand, vendor = vendor, 
                     surplus_avail = surplus_avail, surplus_budget = surplus_budget)  # cost function
    
#    toolbox.register("mate", tools.cxPartialyMatched)
#    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("mate", tools.cxOrdered)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
    
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    # now let's evolute the species!
    for gen in range(NGEN):
        
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.6, mutpb=0.3)
        fits = toolbox.map(toolbox.evaluate, offspring)
        gen_fit = []
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
            #print('generation:' + str(gen))
            gen_fit.append(fit[0])
        population = toolbox.select(offspring, k=len(population))
        # see if converged
        gen_fit = pd.Series(gen_fit)
        trimmed_gen_fit = gen_fit[gen_fit < gen_fit.quantile(0.90)]
        
        # best 2 individuals
        top2 = tools.selBest(population, k=5)
        
        logger.info('''Generation: {0}, MinCost: {1}, MeanCost: {2}, MaxCost: {3}, Trimmed Standard Deviation: {4}, Trimmed Kurtosis: {5},
                    Best 2 Individuals: 
                        {6}, 
                        {7}'''\
                    .format(gen, 
                            np.min(gen_fit), 
                            np.mean(gen_fit), 
                            np.max(gen_fit), 
                            np.std(trimmed_gen_fit), 
                            np.std(trimmed_gen_fit)/np.mean(trimmed_gen_fit), 
                            v[top2[0]], v[top2[1]]))
        #logger.info('Standard Deviation: {0}'.format(str(np.std(gen_fit[gen_fit < gen_fit.quantile(0.90)]))))     
        hof.update(population)
        if np.std(trimmed_gen_fit)/np.mean(trimmed_gen_fit) < 0.03:
            break

        
    # best solution
    vd_perm = v[hof[0]]
    logger.info('Best vendor permutation: {0}'.format(vd_perm))
    output, cost, bought_surplus = run_strategy_vdperm(strategy, vd_perm, cuc_vendor_avail, 
                                                       cuc_vendor_avail_freight, cuc_premium_demand, vendor, 
                                                       surplus_avail, surplus_budget, check_status = False)
    
    if multiprocess:
        pool.close()
    return output, cost, bought_surplus, vd_perm

    
    





def cost_wrapper(perm, v,
                 strategy, cuc_vendor_avail, 
                 cuc_vendor_avail_freight, cuc_premium_demand, vendor, 
                 surplus_avail, surplus_budget):
    """
    wrapper of run_strategy_vdperm() for the correct format to feed to DEAP framework
    """
    #print(perm)
    vd_perm = v[perm]
    _, cost, _= run_strategy_vdperm(strategy, vd_perm, cuc_vendor_avail, 
                                    cuc_vendor_avail_freight, cuc_premium_demand, vendor, 
                                    surplus_avail, surplus_budget, check_status = False)
    #print('cost:' + str(cost))
    return (cost,)








def run_strategy_vdperm(strategy, vd_perm, cuc_vendor_avail, cuc_vendor_avail_freight, 
                        cuc_premium_demand, vendor, surplus_avail, surplus_budget, 
                        check_status = True):
    '''
    worker function for run_all_strategies
    this function will run a specific strategy for one given vendor permutation
    
    Args:
        strategy (function object): strategy to be run
        vd_perm (list of str): vendor permutation to be run
        cuc_vendor_avail (pandas.DataFrame)
        cuc_vendor_avail_freight (pandas.DataFrame)
        cuc_premium_demand (pandas.DataFrame)
        vendor (pandas.DataFrame)
        surplus_avail (pandas.DataFrame)
        surplus_budget (pandas.DataFrame)
        check_status (bool): whether we want to save some current permutation info in order to skip unecessary runs later on (see skip_vd_perm())
    
    '''
    # length of output
    len_output = len(cuc_vendor_avail_freight.qty)        
    
    #initiate output 
    cp_cuc_avail_freight = cuc_vendor_avail_freight.copy()
    cp_cuc_avail = cuc_vendor_avail.copy()
    #cp_cuc_demand = cuc_premium_demand.copy()
    cp_surplus_avail = surplus_avail.copy()
    surplus_result = pd.DataFrame(columns = cp_surplus_avail.columns)
    cp_cuc_avail_freight['output'] = [0 for _ in range(len_output)]
    #[need] column was defined to be cuc need, but here it's the cuc_premium level need
    cp_cuc_avail_freight['cuc_wp_need'] = [1 for _ in range(len_output)]
     
    
     # if there are freight vendors
    if len(vd_perm) > 0:   
                   
        # initiate vendor status, construct Vendor objects and put into a dictionary
        perm_vendor_dict = {}
        for row in vendor.loc[vendor.vendor.isin(vd_perm)].itertuples():
            perm_vendor_dict[row.vendor] = Vendor(freight_type = row.freight_type, freight_value = row.freight_value)
    
        
        # loop over all vendors
        for i in range(len(vd_perm)):
            
            # select items to buy from the vendor
            ind = strategy(cp_cuc_avail_freight, vd_perm[i], perm_vendor_dict[vd_perm[i]])
            
            # update 'output' field to indicate they're bought from the vendor
            # update cuc_wp_need column based on demand_id (once a demand id is filled, all rows associated with this demand_id should be marked 0)
            cp_cuc_avail_freight = update_need(cp_cuc_avail_freight, ind, 'demand_id')
    
            if len(ind)>0:
            # update vendor's freight status
                add_freight_value = cp_cuc_avail_freight.loc[ind, ['cuc', 'qty', 'price']]
                add_freight_value['price'] = add_freight_value.apply(lambda x: x.price/x.qty if re.match('.*_batch$', x.cuc) else x.price, axis = 1)
                perm_vendor_dict[vd_perm[i]].add_value(add_freight_value)    
                  
                
#        ## generate not-buy list
#        
#         everything in cuc_vendor_avail minus the ones we already bought (cp_cuc_avail_freight and cuc_wp_need = 0)
#         already bought matrix
        
        # check freight for each vendor. If not freight, check surplus. Add freight vendor to list
        freight_vendor_generic, freight_vendor_surplus = [], []
        for i in range(len(vd_perm)):
            # vendors that already make freight
            if perm_vendor_dict[vd_perm[i]].freight_reached():
                freight_vendor_generic.append(vd_perm[i])
                
            # if not making freight, and surplus is allowed:
            elif surplus_budget > 0:
                
                # eligible surplus items from this vendor
                items = cp_surplus_avail[(cp_surplus_avail.vendor == vd_perm[i]) & (cp_surplus_avail.cuc_wp_need == 1)]\
                                     .sort_values(by = 'price', ascending = False)
                
                # if possible to make freight with surplus: pick items to buy
                if perm_vendor_dict[vd_perm[i]].make_freight(items, include_current_value = True):
                    
                    # amount freight: check budget, choose from most expensive
                    if perm_vendor_dict[vd_perm[i]].freight_type == 'amount':
                        # surplus parts chosen from this vendor
                        ind = get_until_reach_threshold(items.price, min(surplus_budget, perm_vendor_dict[vd_perm[i]].freight_value - perm_vendor_dict[vd_perm[i]].current_value))
                        # if able to make freight with chosen surplus parts
                        if perm_vendor_dict[vd_perm[i]].make_freight(items.loc[ind, ['price', 'qty']], include_current_value = True):
                            ### add to bought items:
                            cp_surplus_avail, surplus_result, freight_vendor_surplus, surplus_budget = \
                                add_surplus_to_brought_items(cp_surplus_avail, ind, surplus_result, freight_vendor_surplus, vd_perm, surplus_budget)
                            
                    # qty freight: choose from cheapest, check budget
                    elif perm_vendor_dict[vd_perm[i]].freight_type == 'qty':
                        # quantity needed to reach freight
                        qty_need = perm_vendor_dict[vd_perm[i]].freight_value - perm_vendor_dict[vd_perm[i]].current_value
                        # get the cheapest items from item list
                        ind = items.price.index[-qty_need:]
                        # if amount smaller than surplus budget
                        if sum(items.loc[ind, 'price']) <= surplus_budget:
                            #add to bought items
                            cp_surplus_avail, surplus_result, freight_vendor_surplus, surplus_budget = \
                                add_surplus_to_brought_items(cp_surplus_avail, ind, surplus_result, freight_vendor_surplus, vd_perm, surplus_budget)

            
            # get combined freight vendor list
            freight_vendor = freight_vendor_generic + freight_vendor_surplus
            surplus_result = surplus_result.reset_index(drop = True)
        
        # items that are bought from freight vendor
        # filter by output = 1: meaning that item is actually selected from the vendor
        # in some strategies, might not select everything a vendor provide 
        filled_demand_freight = cp_cuc_avail_freight.loc[(cp_cuc_avail_freight.output == 1) & \
                                                           cp_cuc_avail_freight.vendor.isin(freight_vendor)]
        filled_demand_freight = filled_demand_freight.rename(columns = {'price': 'price_freight'})
        filled_demand_id = np.unique(filled_demand_freight['demand_id'])
    
        # cucs that are not bought yet
        cuc_left = cp_cuc_avail.loc[~cp_cuc_avail.demand_id.isin(filled_demand_id)]
        cuc_left['output'] = 0
        cuc_left['cuc_wp_need'] = 1
    
    # if there's no freight vendor
    else:        
        cuc_left = cp_cuc_avail.copy()
        cuc_left['output'] = 0
        cuc_left['cuc_wp_need'] = 1        
        freight_vendor = []
        filled_demand_freight = cp_cuc_avail_freight # no records
        filled_demand_freight = filled_demand_freight.rename(columns = {'price': 'price_freight'})
        
    ## calculate new cost for each cuc, equal to part cost + shipping cost 
    # flag freight vendors
    cuc_left['freight_vendor'] = cuc_left['vendor'].isin(freight_vendor)
    # shipping = 15*qty
    cuc_left['price_single'] = cuc_left.apply(lambda x: x.price + np.where(x.freight_vendor, 0, 15*x.qty), axis = 1)
    
    # single piece price < willingness or [freight vendor batch items]
    filled_demand_single = cuc_left[(cuc_left.price_single < cuc_left.willingness_orig) | \
                                    ((cuc_left.price == cuc_left.price_single) & (cuc_left.cuc.str.endswith('batch')))]\
        .groupby(['demand_id'])\
        .apply(lambda x: x[x.price_single == x.price_single.min()].iloc[0]) # the first row with smallest new_price
    

    # ----------------------------------------------------------
    ### up till now, we have bought everything possible.
    
    ### retrieve bought items from 2 steps
    ### and paste back to cuc demand / cuc avail   
#    # calculate cost
#    # cuc | bp | premium | vendor (vendor to fill this demand from)| price | cost (contribution to cost function)
#    cp_cuc_demand = merge_cuc_demand(cp_cuc_demand, filled_demand_freight, filled_demand_single)
#    # now calculate cost
#    # cost = part cost + opportunity cost (premium of the part that doesn't buy) + time cost (not implemented)
#    cp_cuc_demand['cost'] = np.where(pd.isnull(cp_cuc_demand.price),\
#                                     cp_cuc_demand.premium,
#                                     cp_cuc_demand.price - cp_cuc_demand.bp)
#    cost = cp_cuc_demand['cost'].sum()
        
    # output vector
    # demand_id | vendor | output (binary)
    # same format as cuc_vendor_avail_w_output
    cp_cuc_avail = merge_cuc_avail(cp_cuc_avail, filled_demand_freight, filled_demand_single)
    
    # calculate cost
    shipping = shipping_cost_by_vendor(cp_cuc_avail, vendor)
    cost = part_and_opportunity_cost(cp_cuc_avail, cuc_premium_demand) + shipping.shipping.sum()
    
    
    
    ## update each vendor: remove surplus if freight is reached with reassigned items from other vendors
    # Di 10/04/2018: is this part really necessary? If a vendor couldn make freight in the first round of allocation, that means there should be no more parts that can be allocated to it. 
    # Think this through before doing anything.
    new_surplus_result = pd.DataFrame(columns = surplus_result.columns)
    for vs in np.unique(surplus_result.vendor):
        # if freight type is 'amount'
        if perm_vendor_dict[vs].freight_type == 'amount':
            # current total value OF NEEDED PARTS bought from this vendor
            ttl_value = sum(cp_cuc_avail.price[(cp_cuc_avail.output == 1) & (cp_cuc_avail.vendor == vs)])
            new_surplus_value = max(0, perm_vendor_dict[vs].freight_value - ttl_value)
            # get new 
            ind = get_until_reach_threshold(surplus_result.sort_values('price').loc[surplus_result.vendor == vs, 'price'], new_surplus_value)
            new_surplus_result = new_surplus_result.append(surplus_result.loc[ind])
        # if freight type is 'qty'
        if perm_vendor_dict[vs].freight_type == 'qty':
            ttl_value = sum(cp_cuc_avail.qty[(cp_cuc_avail.output == 1) & (cp_cuc_avail.vendor == vs)])
            new_surplus_value = max(0, perm_vendor_dict[vs].freight_value - ttl_value)
            ind = surplus_result.sort_values('price').index[0:ttl_value]
            new_surplus_result = new_surplus_result.append(surplus_result.loc[ind])

    new_surplus_result.surplus_id = new_surplus_result['surplus_id'].astype(int)
    
    # if we want to record status of current vendor permutation: 
    if check_status:    
    #---------------------------------------
    # this session is to eliminate the average run time
    # update freight_status
        if len(vd_perm) > 0:
            checked_status = {}
            # current freight_making vendors (not including surplus freight vendors)
            # e.g. if vendor perm is ACEBFDG, and ACEF made freight, this will get [ACEBF]
            checked_status['checked_vendor'] = vd_perm[:vd_perm.index(freight_vendor_generic[-1])+1]
            # with current checked vendors, what vendors are still freight-possible?
            checked_demand = cp_cuc_avail_freight.loc[(cp_cuc_avail_freight.output == 1) & \
                                                      cp_cuc_avail_freight.vendor.isin(checked_status['checked_vendor'])]
            cuc_left_2 = cp_cuc_avail.loc[~cp_cuc_avail.demand_id.isin(checked_demand['demand_id'])]
            # e.g.cont'd. check if D and G are freight possible
            remain_freight_poss = vendor_freight_possibility(cuc_left_2, vendor[~vendor.vendor.isin(freight_vendor_generic)])
            checked_status['freight_possible'] = list(remain_freight_poss.loc[remain_freight_poss.possibility == 1, 'vendor'])
            
        else:
            checked_status = {}       
            
        return cp_cuc_avail['output'], cost, checked_status, new_surplus_result.drop('cuc_wp_need', axis = 1)
    
    return cp_cuc_avail['output'], cost, new_surplus_result.drop('cuc_wp_need', axis = 1)
    




def add_surplus_to_brought_items(cp_surplus_avail, ind, surplus_result, freight_vendor_surplus, vd_perm, surplus_budget):
    #add to bought items
    # update surplus item need 
    cp_surplus_avail = update_need(cp_surplus_avail, ind, 'surplus_id')
    # surplus result in cp_cuc_avail_w_output format
    surplus_result = surplus_result.append(cp_surplus_avail.loc[ind])
    # update freight_vendor
    freight_vendor_surplus.append(vd_perm[i])
    # update surplus budget
    surplus_budget = surplus_budget - sum(items.loc[ind, 'price'])

    return(cp_surplus_avail, surplus_result, freight_vendor_surplus, surplus_budget)



def update_need(df, ind, id_col):
    """update 'output' and 'cuc_wp_need' to indicate they're bought from the vendor
    
    Args:
        df (pandas.DataFram): id_col | output | cuc_wp_need
        ind (index object): indices of the rows (demand_id | vendor) to be bought
        id_col (list of str): link between output and cuc_wp_need: output is by demand_id | vendor, and cuc_wp_need should be updated by demand_id. Then id_col in this case is 'demand_id'
    """
    # update 'output' field to indicate they're bought from the vendor
    df.loc[ind, 'output'] = 1      
    
    if len(ind)>0:
    
        # update cuc_wp_need column based on demand_id
        new_fullfill = df.loc[ind, [id_col, 'output']]
        new_fullfill = new_fullfill.rename(columns = {'output': 'newfill'})
        
        df = df.merge(new_fullfill, how = 'left', on = id_col)        
        df.cuc_wp_need = np.where(pd.isnull(df.newfill), \
                                                    df.cuc_wp_need, \
                                                    df.cuc_wp_need - df.newfill)
        df = df.drop('newfill', axis = 1)
        
    return df









def merge_cuc_demand(base_df, filled_demand_freight, filled_demand_single):
    '''
    this function merges cp_cuc_demand with filled_demand_freight and filled_demand_single
    it's called in run_strategy_vdperm()
    
    Args:
        filled_demand_freight (pandas.DataFrame): demand filled by freight vendors
        filled_demand_single (pandas.DataFrame): demand filled by single piecing (or add-on to freight vendors) after freight vendor are chosen
    '''
    
    # filled_demand_freight, renaming variables to avoid name collapse in mergine
    if len(filled_demand_freight) > 0:
        filled_demand_freight = filled_demand_freight[['demand_id', 'vendor', 'price_freight']]
        filled_demand_freight = filled_demand_freight.rename(columns = {'vendor': 'vendor_freight'})
    else:
        filled_demand_freight = pd.DataFrame({'demand_id':[], 'vendor_freight':[], 'price_freight': []})
        
    # filled_demand_single
    if len(filled_demand_single) > 0:
        filled_demand_single = filled_demand_single[['demand_id', 'vendor', 'price_single']]
        filled_demand_single = filled_demand_single.rename(columns = {'vendor': 'vendor_single'})
    else:
        filled_demand_single = pd.DataFrame({'demand_id':[], 'vendor_single':[], 'price_single': []})
    
    # merging
    base_df = \
    base_df.merge(filled_demand_freight[['demand_id', 'vendor_freight', 'price_freight']].reset_index(drop = True), \
                             how = 'left')\
                 .merge(filled_demand_single[['demand_id', 'vendor_single', 'price_single']].reset_index(drop = True), \
                             how = 'left')
    
    base_df['vendor'] = np.where(pd.isnull(base_df.vendor_single), base_df.vendor_freight, base_df.vendor_single)
    base_df['price'] = np.where(pd.isnull(base_df.price_single), base_df.price_freight, base_df.price_single)
    
    base_df = base_df.drop(['vendor_freight', 'vendor_single', 'price_freight', 'price_single'], axis = 1)

    return base_df





def merge_cuc_avail(base_df, filled_demand_freight, filled_demand_single):
    '''
    this function merges cp_cuc_avail with filled_demand_freight and filled_demand_single
    returning dataframe has an extra column: output, which is a binary column indicating whether a row (a supply to a demand_id) is selected
    it's called in run_strategy_vdperm()
    
    Args:
        filled_demand_freight: demand filled by freight vendors
        filled_demand_single: demand filled by single piecing (or add-on to freight vendors) after freight vendor are chosen

    '''
    
    # filled_demand_freight
    if len(filled_demand_freight) > 0:
        filled_demand_freight = filled_demand_freight[['demand_id', 'vendor', 'price_freight']]
        #filled_demand_freight = filled_demand_freight.rename(columns = {'vendor': 'vendor_freight'})
    else:
        filled_demand_freight = pd.DataFrame({'demand_id':[], 'vendor':[], 'price_freight': []})
        
    # filled_demand_single
    if len(filled_demand_single) > 0:
        filled_demand_single = filled_demand_single[['demand_id', 'vendor', 'price_single']]
        #filled_demand_single = filled_demand_single.rename(columns = {'vendor': 'vendor_single'})
    else:
        filled_demand_single = pd.DataFrame({'demand_id':[], 'vendor':[], 'price_single': []})
    
    # merging
    base_df = \
    base_df.merge(filled_demand_freight[['demand_id', 'vendor', 'price_freight']].reset_index(drop = True), \
                             how = 'left')\
                 .merge(filled_demand_single[['demand_id', 'vendor', 'price_single']].reset_index(drop = True), \
                             how = 'left')
    
    base_df['output'] = np.where(pd.isnull(base_df.price_freight) & pd.isnull(base_df.price_single), 0, 1)

    base_df = base_df.drop(['price_freight', 'price_single'], axis = 1)

    return base_df









