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
pool = ['11', '137', '171', '234', '311']

# to get every user this code needs to be used
# pool = []
# for user_id in service_provider_handler.local_data[provider_id]:
#    pool.append(user_id)

# This time they would like to measure fairness metrics for people that are female and disabled
attribute_names = ['gender', "disabled"]
attribute_values = ['female', "True"]
# Assertion len(attribute_names) == len(attribute_values)

# They create a handler with their API authentication key
if mpc.pid == 0:
    fairness_handler = MultipartyFairnessMeasurementMPYC(api_key=provider_id, data_handler=third_party_handler)
else:
    fairness_handler = MultipartyFairnessMeasurementMPYC(api_key=provider_id, data_handler=service_provider_handler)


async def main():
    await mpc.start()

    diversity = await fairness_handler.measure_pool_diversity(pool, attribute_names, attribute_values)

    if mpc.pid == 0:
        print ("THIRD-PARTY OUTPUT: the fraction of female disabled candidates in the pool is: %f" % (diversity))
    else:
        print ("SERVICE-PROVIDER OUTPUT: the fraction of female disabled candidates in the pool is: %f" % (diversity))

    exposure = await fairness_handler.measure_group_exposure(pool, attribute_names, attribute_values, 
                                            browsing_model = [0.5, 0.2, 0.1, 0.1, 0.1])

    if mpc.pid == 0:
        print ("THIRD-PARTY OUTPUT: The exposure of female disabled candidates in the ranking is: %f" % (exposure))
    else:
        print ("SERVICE-PROVIDER OUTPUT: The exposure of female disabled candidates in the ranking is: %f" % (exposure))

    pool_stage = [("11", True, True), ("137", True, True), ("171", True, False), ("234", True, True), ("311", True, True)]
    demographic_parity = await fairness_handler.measure_accept_rate(pool, pool_stage, attribute_names, attribute_values)

    if mpc.pid == 0:
        print ("THIRD-PARTY OUTPUT: The demographic parity score for female disabled candidates is: %f" % (demographic_parity))
    else:
        print ("SERVICE-PROVIDER OUTPUT: The demographic parity score for female disabled candidates is: %f" % (demographic_parity))

    pool_stage = [("11", True, False), ("137", True, True), ("171", False, False), ("234", False, True), ("311", True, True)]
    equal_opportunity = await fairness_handler.measure_accept_rate(pool, pool_stage, attribute_names, attribute_values)

    if mpc.pid == 0:
        print ("THIRD-PARTY OUTPUT: The equal opportunity score for female disabled candidates is: %f" % (equal_opportunity))
    else:
        print ("SERVICE-PROVIDER OUTPUT: The equal opportunity score for female disabled candidates is: %f" % (equal_opportunity))

    await mpc.shutdown()

mpc.run(main())
