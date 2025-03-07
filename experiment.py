import yaml

real_experiment = open('real-experiment.yml', 'r')
real_experiment = yaml.load(real_experiment, Loader=yaml.FullLoader)

treatments = real_experiment['treatments']
factors = real_experiment['factors']
factorTypes = real_experiment['factorTypes']

def get_factor_by_id(factors, factor_id):
    for factor in factors:
        if factor['_id'] == factor_id:
            return factor
    return None

def get_factor_type_by_id(factorTypes, factor_type_id):
    for factorType in factorTypes:
        if factorType['_id'] == factor_type_id:
            return factorType
    return None

for treatment in treatments:
    print(treatment['name'])
    print()
    for factorId in treatment['factorIds']:
        factor = get_factor_by_id(factors, factorId)
        factorType = get_factor_type_by_id(factorTypes, factor['factorTypeId'])
        if factorType:
            print(factorType['name'], ":", factor['value'] , f"({factorType['description']})")
    print('---')
