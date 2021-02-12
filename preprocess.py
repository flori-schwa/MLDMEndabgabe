import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Callable, Any, Optional, Set

import pandas as pd

import arff

CLASS_OVER_50K = '>50K'
CLASS_UNDER_EQ_50K = "<=50K"


def transform_func(base_attr_name: str, new_attr: arff.Attribute):
    def decorate(func):
        setattr(func, 'base_attr_name', base_attr_name)
        setattr(func, 'new_attr', new_attr)
        return func

    return decorate


def quantil(data, p: float, by: Callable[[Any], float]) -> Optional[float]:
    if p <= 0 or p >= 1:
        return None

    n = len(data)
    np = n * p

    if math.floor(np) == math.ceil(np):
        return (by(data[int(np - 1)]) + by(data[int(np)])) / 2
    else:
        return by(data[math.ceil(np - 1)])


def median(data, by: Callable[[Any], float]) -> Optional[float]:
    return quantil(data, 0.5, by)


def analyze_countries(census_data: pd.DataFrame) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    over_50k: Dict[str, int] = {}
    under_eq_50k: Dict[str, int] = {}

    country_count: Dict[str, int] = {}

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

    percentages: Dict[str, float] = {}
    unique_countries: Set[str] = set([key for key in over_50k]).union(set([key for key in under_eq_50k]))

    for country in unique_countries:
        num: float = over_50k[country] if country in over_50k else 0
        denum: float = num + (under_eq_50k[country] if country in under_eq_50k else 0)

        percentages[country] = 0 if denum == 0 else num / denum

    return percentages, sorted(percentages.items(), key=lambda x: x[1])


class TransformContext:
    def __init__(self):
        self.sorted_countries = None
        self.country_percentages: Dict[str, float] = None
        self.country_median: float = 0
        self.raw_data: pd.DataFrame = None
        self.base_attributes: List[arff.Attribute] = None
        self.attribute_transforms: Dict[str, Tuple[arff.Attribute, Callable[['TransformContext', Any], Any]]] = {}

    def drop_attribute(self, base_attr_name: str):
        self.attribute_transforms[base_attr_name] = (None, None)

    def drop_attributes(self, *argv):
        for arg in argv:
            self.drop_attribute(arg)

    # noinspection PyUnresolvedReferences
    def transform_attribute(self, transform_func: Callable[['TransformContext', Any], Any]):
        self.attribute_transforms[transform_func.base_attr_name] = (transform_func.new_attr, transform_func)

    def initialize_transformation(self, raw_data: pd.DataFrame, base_attributes: List[arff.Attribute]):
        self.raw_data = raw_data
        self.base_attributes = base_attributes

    def transform_data(self, new_relation_name: str, arff_out: str, csv_out: str) -> List[arff.Attribute]:
        data_transformed: List[List[Any]] = []

        for _, row in self.raw_data.iterrows():
            new_row = []

            for attr in self.base_attributes:
                if attr.name in self.attribute_transforms:
                    new_attr_type, transform_func = self.attribute_transforms[attr.name]

                    if new_attr_type is None or transform_func is None:
                        continue  # Drop Attribute

                    transformed = transform_func(self, row[attr.name])

                    if not new_attr_type.is_valid(transformed):
                        print(f"Invalid data returned for transformed attribute {new_attr_type.name}: {transformed}")
                        return None

                    new_row.append(transformed)
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

        return new_attributes

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


@transform_func('class', arff.NominalAttribute('target', ['1', '0']))
def class_to_target(_: TransformContext, clazz: str) -> str:
    return '0' if clazz == CLASS_OVER_50K else '1'


import Florian, Frank, Phillipp

if __name__ == "__main__":
    census_raw, census_attributes = arff.parse_arff_file("adult_train.arff")
    tc: TransformContext = TransformContext()

    if not tc.load():
        tc.calc(census_raw)
        tc.save()

    print("Transforming data...")

    tc.initialize_transformation(census_raw, census_attributes)
    tc.drop_attributes('fnlwgt', 'race', 'capital-loss', 'education', 'relationship', 'capital-loss')
    tc.transform_attribute(class_to_target)

    tc.transform_attribute(Florian.transform_cntry_percentage)
    # tc.transform_attribute(Florian.transform_education_hs_col_grad)
    tc.transform_attribute(Florian.transform_hrs_per_week)

    tc.transform_attribute(Frank.transform_age_c)
    tc.transform_attribute(Frank.transform_capital_gain_bin)
    # tc.transform_attribute(Frank.transform_capital_loss_bin)

    tc.transform_attribute(Phillipp.transform_marital_status)
    # tc.transform_attribute(Phillipp.transform_relationship)

    tc.transform_data('census_transformed', 'out/data_transformed.arff', 'out/data_transformed.csv')
