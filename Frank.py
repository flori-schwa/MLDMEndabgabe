from arff import NumericAttribute, NominalAttribute
from preprocess import TransformContext, transform_func


@transform_func('age', NumericAttribute('ageC'))
def transform_age_c(_: TransformContext, age: int) -> int:
    if age is None:
        return 0
    if age < 18:
        return 1
    elif age < 35:
        return 2
    elif age < 55:
        return 3
    return 4


@transform_func('capital-gain', NominalAttribute('capital-gainB', ["Yes", "No"]))
def transform_capital_gain_bin(_: TransformContext, cg: float) -> str:
    return "Yes" if cg > 5000 else "No"


@transform_func('capital-loss', NumericAttribute('capital-lossB'))
def transform_capital_loss_bin(_: TransformContext, cl: float) -> int:
    return 1 if cl > 0 else 0
