import arff_parser
import pandas as pd
import math
from pathlib import Path
import json


class TransformContext:

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

        return True

    def save(self):
        print("Saving TransformContext...")
        Path('transform_data').mkdir(parents=True, exist_ok=True)


def combine_wife_husband(relationship):
    if relationship is None:
        return 'Unknown'
    if relationship == '?':
        return 'Unknown'
    if relationship == 'Husband':
        return 'Married'
    if relationship == 'Wife':
        return 'Married'
    if relationship == 'Own-child':
        return 'Own-child'
    if relationship == 'Not-in-family':
        return 'Not-in-family'
    if relationship == 'Other-relative':
        return 'Other-relative'
    if relationship == 'Unmarried':
        return 'Unmarried'


def combine_marital_status(marital_status):
    if marital_status is None:
        return 'Unknown'
    if marital_status == '?':
        return 'Unknown'
    if marital_status == 'Married-civ-spouse':
        return 'Married-spouse'
    if marital_status == 'Married-spouse-absent':
        return 'Married-spouse'
    if marital_status == 'Married-af-spouse':
        return 'Married-spouse'
    if marital_status == 'Divorced':
        return 'Unmarried'
    if marital_status == 'Widowed':
        return 'Unmarried'
    if marital_status == 'Never-married':
        return 'Never-married'
    if marital_status == 'Separated':
        return 'Separated'


if __name__ == "__main__":
    census_raw = arff_parser.parse_arff_file("adult_train.arff")
    tc: TransformContext = TransformContext()

    if not tc.load():
        tc.save()

    print("Transforming data...")

    data_transformed = []

    for _, row in census_raw.iterrows():
        data_transformed.append([
            row['age'],
            row['workclass'],
            row['fnlwgt'],
            row['education'],
            row['education-num'],
            combine_marital_status(row['marital-status']),
            row['occupation'],
            combine_wife_husband(row['relationship']),
            row['race'],
            row['sex'],
            row['capital-gain'],
            row['capital-loss'],
            row['hours-per-week'],
            row['native-country'],
            row['class']
        ])

    with open('data_transformed.csv', 'w') as f:
        for row in data_transformed:
            f.write((", ".join([('?' if x is None else str(x)) for x in row])) + '\n')
