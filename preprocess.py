import arff_parser
import pandas as pd
import math
from pathlib import Path
import json

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
        return 'Unknown'
    if workclass == 'Private':
        return workclass
    if workclass in ['Self-emp-not-inc', 'Self-emp-inc']:
        return 'Self-emp'
    if workclass in ['Federal-gov', 'Local-gov', 'State-gov']:
        return 'Gov'
    return 'Unpaid'


if __name__ == "__main__":
    census_raw = arff_parser.parse_arff_file("adult.arff")
    tc: TransformContext = TransformContext()

    if not tc.load():
        tc.calc(census_raw)
        tc.save()

    print("Transforming data...")

    data_transformed = []

    for _, row in census_raw.iterrows():
        data_transformed.append([
            row['age'],
            transform_workclass(row['workclass']),
            row['fnlwgt'],
            transform_education_num(row['education-num']),
            row['marital-status'],
            row['occupation'],
            row['relationship'],
            row['race'],
            row['sex'],
            row['capital-gain'],
            row['capital-loss'],
            row['hours-per-week'],
            transform_country(tc, row['native-country']),
            row['class']
        ])

    with open('out/data_transformed.csv', 'w') as f:
        for row in data_transformed:
            f.write((", ".join([('?' if x is None else str(x)) for x in row])) + '\n')
