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
        country = row['native-country']
        country = "None" if country is None else country

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

    def calc(self, census_data: pd.DataFrame):
        print("TransformContext not found, calculating data...")
        self.country_percentages, self.sorted_countries = analyze_countries(census_data)
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


def transform_countries_rich_poor(census_data: pd.DataFrame, tc: TransformContext):
    new_data = []
    median_perc = median(tc.sorted_countries, by=lambda x: x[1])

    for idx, row in census_data.iterrows():
        cntry = row['native-country'];
        cntry = "None" if cntry is None else cntry

        new_data.append([
            row['age'],
            row['workclass'],
            row['fnlwgt'],
            row['education'],
            row['education-num'],
            row['marital-status'],
            row['occupation'],
            row['relationship'],
            row['race'],
            row['sex'],
            row['capital-gain'],
            row['capital-loss'],
            row['hours-per-week'],
            1 if tc.country_percentages[cntry] >= median_perc else 0,
            row['class']
        ])

    return new_data


if __name__ == "__main__":
    census_raw = arff_parser.parse_arff_file("adult.arff")
    tc: TransformContext = TransformContext()

    if not tc.load():
        tc.calc(census_raw)
        tc.save()

    print("Transforming data...")

    countries_transformed = transform_countries_rich_poor(census_raw, tc)

    with open('out/countries_transformed.csv', 'w') as f:
        for row in countries_transformed:
            f.write((", ".join([('?' if x is None else str(x)) for x in row])) + '\n')
