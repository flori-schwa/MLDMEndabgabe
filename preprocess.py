import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any

import pandas as pd

import arff

CLASS_OVER_50K = '>50K'
CLASS_UNDER_EQ_50K = "<=50K"


def quantil(data, p, by):
    if p < 0 or p > 1:
        return None

    n = len(data)
    np = n * p

    if math.floor(np) == math.ceil(np):
        return (by(data[int(np - 1)]) + by(data[int(np)])) / 2
    else:
        return by(data[math.ceil(np - 1)])


def median(data, by):
    return quantil(data, 0.5, by)


def analyze_countries(census_data: pd.DataFrame):
    over_50k = {}
    under_eq_50k = {}

    country_count = {}

    for idx, row in census_data.iterrows():
        country = str(row['native-country'])  # str() to convert `None` to `"None"`

        if country in country_count:
            country_count[country] += 1
        else:
            country_count[country] = 1

        if row['class'] == CLASS_OVER_50K:
            if country in over_50k:
                over_50k[country] += 1
            else:
                over_50k[country] = 1
        else:
            if country in under_eq_50k:
                under_eq_50k[country] += 1
            else:
                under_eq_50k[country] = 1

    percentages = {}
    unique_countries = set([key for key in over_50k]).union(set([key for key in under_eq_50k]))

    for country in unique_countries:
        num = over_50k[country] if country in over_50k else 0
        denum = num + (under_eq_50k[country] if country in under_eq_50k else 0)
        percentages[country] = 0 if denum == 0 else num / denum

    return percentages, sorted(percentages.items(), key=lambda x: x[1])


class TransformContext:
    def __init__(self):
        self.sorted_countries = None
        self.country_percentages = None
        self.country_median = 0
        self.raw_data: pd.DataFrame = None
        self.base_attributes: List[arff.Attribute] = None
        self.attribute_transforms: Dict[str, Tuple[arff.Attribute, Callable[[Any], Any]]] = {}

    def drop_attribute(self, base_attr_name: str):
        self.add_attribute_transformation(base_attr_name, None, None)

    def drop_attributes(self, *argv):
        for arg in argv:
            self.drop_attribute(arg)

    def add_attribute_transformation(self, base_attr_name: str, new_attr_type: arff.Attribute,
                                     transform_func: Callable[[Any], Any]):
        self.attribute_transforms[base_attr_name] = (new_attr_type, transform_func)

    def initialize_transformation(self, raw_data: pd.DataFrame, base_attributes: List[arff.Attribute]):
        self.raw_data = raw_data
        self.base_attributes = base_attributes

    def transform_data(self, new_relation_name: str, arff_out: str, csv_out: str):
        data_transformed: List[List[Any]] = []

        for _, row in self.raw_data.iterrows():
            new_row = []

            for attr in self.base_attributes:
                if attr.name in self.attribute_transforms:
                    new_attr_type, transform_func = self.attribute_transforms[attr.name]

                    if new_attr_type is None or transform_func is None:
                        continue  # Drop Attribute

                    new_row.append(transform_func(row[attr.name]))
                else:
                    new_row.append(row[attr.name])

            data_transformed.append(new_row)

        new_attributes = []

        for attr in self.base_attributes:
            if attr.name in self.attribute_transforms:
                new_attr_type, transform_func = self.attribute_transforms[attr.name]

                if new_attr_type is None or transform_func is None:
                    continue  # Drop Attribute

                new_attributes.append(new_attr_type)
            else:
                new_attributes.append(attr)

        with open(arff_out, "w") as arff_f:
            arff.write_arff_header(arff_f, new_relation_name, new_attributes)

            with open(csv_out, "w") as csv_f:
                csv_f.write(f"{','.join([new_attr.name for new_attr in new_attributes])}\n")
                arff.write_csv(csv_f, data_transformed)

            arff.write_csv(arff_f, data_transformed)

    def __calc_median(self):
        self.country_median = median(self.sorted_countries, by=lambda x: x[1])

    def calc(self, census_data: pd.DataFrame):
        print("TransformContext not found, calculating data...")
        self.country_percentages, self.sorted_countries = analyze_countries(census_data)
        self.__calc_median()
        pass

    @staticmethod
    def require_file_read(filename):
        path = Path('transform_data') / filename

        if not path.exists():
            raise FileNotFoundError()

        return open(path.resolve(), 'r')

    def load(self):
        print("Loading TransformContext...")
        path: Path = Path('transform_data')

        if not path.is_dir():
            return False

        try:
            with self.require_file_read('country_percentages.json') as f:
                self.country_percentages = json.load(f)

            with self.require_file_read('sorted_countries.json') as f:
                self.sorted_countries = json.load(f)

            self.__calc_median()
        except FileNotFoundError:
            return False

        return True

    def save(self):
        print("Saving TransformContext...")
        Path('transform_data').mkdir(parents=True, exist_ok=True)

        with open('transform_data/country_percentages.json', 'w') as f:
            json.dump(self.country_percentages, f)

        with open('transform_data/sorted_countries.json', 'w') as f:
            json.dump(self.sorted_countries, f)


def transform_country(tc: TransformContext, cntry):
    return 1 if tc.country_percentages[str(cntry)] >= tc.country_median else 0  # str() to convert `None` to `"None"`


def transform_country_regional(cntry):
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


def transform_education_is_grad_college(education_num: int):
    return "Yes" if education_num >= 13 else "No"


def transform_education_hs_col_grad(education_num: int):
    if education_num < 9:
        return "No-HS"
    if education_num == 9:
        return "HS-Grad"
    if education_num < 13:
        return "College"
    return "College-Grad"


def transform_education_num(education_num: int):
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


def transform_workclass(workclass):
    if workclass is None:
        return None
    if workclass == 'Private':
        return workclass
    if workclass in ['Self-emp-not-inc', 'Self-emp-inc']:
        return 'Self-emp'
    if workclass in ['Federal-gov', 'Local-gov', 'State-gov']:
        return 'Gov'
    return 'Unpaid'


def transform_hrs_per_week(x: int):
    return 1 if x >= 40 else 0


def transform_age(age: int):
    if age < 34:
        return 1
    if age < 44:
        return 2
    return 3


if __name__ == "__main__":
    census_raw, census_attributes = arff.parse_arff_file("adult_train.arff")
    tc: TransformContext = TransformContext()

    if not tc.load():
        tc.calc(census_raw)
        tc.save()

    print("Transforming data...")

    tc.initialize_transformation(census_raw, census_attributes)
    tc.drop_attributes('fnlwgt', 'race', 'capital-loss', 'education')
    tc.add_attribute_transformation('workclass',
                                    arff.NominalAttribute('workclass', ['Private', 'Self-emp', 'Gov', 'Unpaid']),
                                    transform_workclass)
    tc.add_attribute_transformation('education-num', arff.NominalAttribute('is-college-grad', ['Yes', 'No']),
                                    transform_education_is_grad_college)

    tc.transform_data('census_transformed', 'out/data_transformed.arff', 'out/data_transformed.csv')
    # data_transformed = []
    #
    # for _, row in census_raw.iterrows():
    #     data_transformed.append([
    #         transform_age(row['age']),
    #         row['workclass'],  # row['workclass'],  # transform_workclass(row['workclass']),
    #         row['fnlwgt'],
    #         row['education'],
    #         row['education-num'],  # transform_education_is_grad_college(row['education-num']), # transform_education_num(row['education-num']), # row['education-num'],  # transform_education_num(row['education-num']),
    #         row['marital-status'],
    #         row['occupation'],
    #         row['relationship'],
    #         row['race'],
    #         row['sex'],
    #         row['capital-gain'],
    #         row['capital-loss'],
    #         row['hours-per-week'],
    #         row['native-country'], # transform_country_regional(row['native-country']),  # transform_country(tc, row['native-country']),
    #         row['class']
    #     ])
    #
    # with open('out/data_transformed.csv', 'w') as f:
    #     for row in data_transformed:
    #         f.write((", ".join([('?' if x is None else str(x)) for x in row])) + '\n')
