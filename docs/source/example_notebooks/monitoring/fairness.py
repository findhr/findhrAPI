from mpyc.runtime import mpc

import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../../src"))
sys.path.append(project_root)

from findhr.monitoring import *

filename_third_party = '../../../data/risk_monitoring/third_party_local_data.csv'
filename_service_provider = '../../../data/risk_monitoring/service_provider_remote_data.csv'

third_party_handler = MultipartyDataHandlerCSV(filename_third_party)
service_provider_handler = ServiceProviderHandlerCSV(filename_service_provider)

# We want to measure data from this provider
provider_id = "346723798"

# We want to measure fairness for the ranking of these 5 users:
pool = ['11', '137', '171', '234', '279']

# to get every user this code needs to be used
# pool = []
# for user_id in service_provider_handler.local_data[provider_id]:
#    pool.append(user_id)

# They would like to measure diversity wrt gender
attribute_name = "gender"

# And measure the prevalence of female candidates
attribute_value = "female"

# They create a handler with their API authentication key
if mpc.pid == 0:
    fairness_handler = MultipartyFairnessMeasurementMPYC(api_key=provider_id, data_handler=third_party_handler)
else:
    fairness_handler = MultipartyFairnessMeasurementMPYC(api_key=provider_id, data_handler=service_provider_handler)


def print_two_party(string):
    if mpc.pid == 0:
        print ("THIRD-PARTY OUTPUT: " + string)
    else:
        print ("SERVICE-PROVIDER OUTPUT: " + string)

async def main():
    await mpc.start()

    # They send this data to the third party, and get the value of the pool diversity fairness metric back:
    diversity = await fairness_handler.measure_pool_diversity(pool, attribute_name, attribute_value)

    # In some cases, they might want to measure conditional diversity -- diversity of the pool from among e.g. qualified candidates.
    # An optional 'conditionals' argument allows them to specify a list of boolean values
    # specifying whether the candidate should be included in the computation.
    # Assertion: len(pool) == len(conditionals)
    diversity = await fairness_handler.measure_pool_diversity(pool, attribute_name, attribute_value,
                                    conditionals=[True, True, True, True, True])

    # This is the fraction of female candidates in the above pool (2/5):
    print_two_party("The fraction of female candidates in the pool is: %f" % (diversity))


    # They might also want to measure the exposure of a given group in a ranking under a selected browsing model.
    # A browsing model is a set of weights indicating how much attention searchers pay to a given ranking position.
    exposure = await fairness_handler.measure_group_exposure(pool, attribute_name, attribute_value,
                                            browsing_model = [0.5, 0.2, 0.1, 0.1, 0.1])

    # This is the fraction of exposure of female candidates in the above ranking, from positions 1 and 5 (0.6):
    print_two_party("The exposure of female candidates in the ranking (manual weights) is: %f" % (exposure))

    # In addition to mannually defining the weights, The browsing model can also be based on predefined models, such as Inverse Log Model and Exponential Decay Model
    exposure = await fairness_handler.measure_group_exposure(pool, attribute_name, attribute_value, browsing_model="inverse_log")
    print_two_party("The exposure of female candidates in the ranking (Inverse Log Model) is: %f" % (exposure))

    exposure = await fairness_handler.measure_group_exposure(pool, attribute_name, attribute_value, browsing_model="exp_decay", browsing_param=0.7)
    print_two_party("The exposure of female candidates in the ranking (Exp Decay Model) is: %f" % (exposure))

    # They might also want to evaluate fairness based on the top-k positions of a ranking.

    # Skew: Measures the difference between the proportion of the group in top-k vs. overall.
    fairness_skew = await fairness_handler.measure_topk_fairness(pool, attribute_name, attribute_value, k=3, method="skew")
    print_two_party("Top-3 fairness of female candidates in the ranking (Skew@k) is: %f" % fairness_skew)

    # Discounted Representation Difference: Captures position-sensitive differences in group presence in top-k.
    fairness_discounted_diff = await fairness_handler.measure_topk_fairness(pool, attribute_name, attribute_value, k=3, method="discounted_rep_diff")
    print_two_party("Top-3 fairness of female candidates in the ranking (Discounted Rep. Diff@k) is: %f" % fairness_discounted_diff)



    # To measure the accept rate of a given group for demographic parity
    # For this a pool_stage list (user_id: string, targeted: 1, selected: bool) needs to be provided
    # For each candidate, the first bool (targeted) is set to 1 to include all potential candidates;
    # and the second bool (selected) indicates the decisions (e.g., final interview or recruitment decisions)
    pool_stage = [("11", True, True), ("137", True, True), ("171", True, False), ("234", True, False), ("279", True, True)]
    demographic_parity = await fairness_handler.measure_accept_rate(pool, pool_stage, attribute_name, attribute_value)

    # This is the proportion of qualified female individuals who were correctly selected:
    print_two_party("The demographic parity measurement of female candidates in the pool is: %f" % (demographic_parity))

    # To measure the proportion of qualified individuals who were correctly selected.
    # For this a pool_stage list (user_id: string, targeted: bool, selected: bool) needs to be provided
    # For each candidate, the first bool (targeted) is set to whether the candidate is qualified or not, e.g., based on evaluations by a group of experts;
    # and the second bool (selected) indicates the decisions (e.g., final interview or recruitment decisions)
    pool_stage = [("11", True, False), ("137", True, True), ("171", False, False), ("234", False, True), ("279", True, True)]
    equal_opportunity = await fairness_handler.measure_accept_rate(pool, pool_stage, attribute_name, attribute_value)

    # This is the proportion of qualified female individuals who were correctly selected:
    print_two_party("The equal opportunity measurement of female candidates in the pool is: %f" % (equal_opportunity))

    await mpc.shutdown()

mpc.run(main())
