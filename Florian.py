from arff import NumericAttribute, NominalAttribute
from preprocess import TransformContext, transform_func


@transform_func('relationship', NominalAttribute("own-child", ["Yes", "No"]))
def transform_relationship(_: TransformContext, rel):
    return "Yes" if rel == "Own-child" else "No"


@transform_func('native-country', NumericAttribute("rich-country"))
def transform_cntry_rich_poor(tc: TransformContext, cntry) -> int:
    return 1 if tc.country_percentages[str(cntry)] >= tc.country_median else 0  # str() to convert `None` to `"None"`


@transform_func('native-country', NumericAttribute("cntry-class-perc"))
def transform_cntry_percentage(tc: TransformContext, cntry: str) -> float:
    return tc.country_percentages["None" if cntry is None else cntry]


@transform_func('native-country', NominalAttribute("native-region",
                                                   ["West-Europe", "East-Europe", "North-America", "Latin-America",
                                                    "Outlying-US",
                                                    "Middle-East", "Asia"]))
def transform_cntry_regional(_: TransformContext, cntry: str) -> str:
    if cntry is None:
        return None
    if cntry in ["Holand-Netherlands", "Scotland", "Portugal", "Germany", "Ireland", "Italy", "England", "France"]:
        return "West-Europe"
    elif cntry in ["Poland", "Hungary", "Yugoslavia", "Greece"]:
        return "East-Europe"
    elif cntry in ["United-States", "Canada"]:
        return "North-America"
    elif cntry in ["Guatemala", "Columbia", "Dominican-Republic", "Mexico", "Nicaragua", "El-Salvador",
                   "Trinadad&Tobago", "Peru", "Honduras", "Puerto-Rico", "Haiti", "Ecuador", "Jamaica", "Cuba"]:
        return "Latin-America"
    elif cntry in ["Outlying-US(Guam-USVI-etc)"]:
        return "Outlying-US"
    elif cntry in ["Iran"]:
        return "Middle-East"
    elif cntry in ["Vietnam", "Laos", "Thailand", "Hong", "Philippines", "China", "Cambodia", "Japan", "Taiwan",
                   "India"]:
        return "Asia"
    else:
        return None


"""
1: Preschool
2: 1st-4th
3: 5th-6th
4: 7th-8th
5: 9th
6: 10th
7: 11th
8: 12th
9: Hs-grad
10: Some-college
11: Assoc-voc
12: Assoc-acdm
13: Bachelors
14: Masters
15: Prof-school
16: Doctorate

-----

1: Preschool
2: Elementary School
3: Middle School
4: High School
5: College
6: Bachelors
7: Masters
8: Prof-school
9: Doctorate
"""


@transform_func('education-num', NumericAttribute("college-grad"))
def transform_education_is_grad_college(_: TransformContext, education_num: int) -> int:
    return 1 if education_num >= 13 else 0


@transform_func('education-num', NominalAttribute("education", ["No-HS", "HS-Grad", "College", "College-Grad"]))
def transform_education_hs_col_grad(_: TransformContext, education_num: int):
    if education_num < 9:
        return "No-HS"
    if education_num == 9:
        return "HS-Grad"
    if education_num < 13:
        return "College"
    return "College-Grad"


@transform_func('education-num', NumericAttribute("education"))
def transform_education_num(_: TransformContext, education_num: int):
    if education_num == 1:
        return 1  # Preschool
    elif education_num == 2:
        return 2  # Elementary School
    elif education_num <= 4:
        return 3  # Middle School
    elif education_num <= 9:
        return 4  # High School
    elif education_num <= 12:
        return 5  # College
    return education_num - 7  # Bachelors - Doctorate


@transform_func('workclass', NominalAttribute("workclass", ["Private", "Self-emp", "Gov", "Unpaid"]))
def transform_workclass(_: TransformContext, workclass):
    if workclass is None:
        return None
    if workclass == 'Private':
        return workclass
    if workclass in ['Self-emp-not-inc', 'Self-emp-inc']:
        return 'Self-emp'
    if workclass in ['Federal-gov', 'Local-gov', 'State-gov']:
        return 'Gov'
    return 'Unpaid'


@transform_func('hrs-per-week', NumericAttribute("hrs-per-week-gte-40"))
def transform_hrs_per_week(_: TransformContext, x: int):
    return 1 if x >= 40 else 0


@transform_func('age', NumericAttribute("age-group"))
def transform_age(_: TransformContext, age: int):
    if age < 34:
        return 1
    if age < 44:
        return 2
    return 3
