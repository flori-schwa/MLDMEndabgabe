from arff import NominalAttribute
from preprocess import TransformContext, transform_func


@transform_func('relationship', NominalAttribute('relationship',
                                                 ['Unknown', 'Married', 'Own-child', 'Not-in-family',
                                                  'Other-relative', 'Unmarried']))
def transform_relationship(_: TransformContext, relationship: str) -> str:
    if relationship is None or relationship == '?':
        return 'Unknown'
    elif relationship in ['Husband', 'Wife']:
        return 'Married'
    return relationship


@transform_func('marital-status', NominalAttribute('marital-status',
                                                   ['Unknown', 'Married-spouse', 'Unmarried', 'Never-married',
                                                    'Separated']))
def transform_marital_status(_: TransformContext, marital_status: str) -> str:
    if marital_status is None or marital_status == '?':
        return 'Unknown'
    elif marital_status in ['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent']:
        return 'Married-spouse'
    elif marital_status in ['Divorced', 'Widowed']:
        return 'Unmarried'
    return marital_status
