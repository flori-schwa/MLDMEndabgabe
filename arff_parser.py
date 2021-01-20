import datetime
from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

from pandas import DataFrame

# region Attribute Class Definitions

T = TypeVar('T')


class Attribute(ABC, Generic[T]):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_valid(self, x: str) -> bool:
        ...

    @abstractmethod
    def parse(self, x: str) -> T:
        ...


class NumericAttribute(Attribute[float]):
    def is_valid(self, x: str) -> bool:
        try:
            self.parse(x)
            return True
        except ValueError:
            return False

    def parse(self, x: str) -> float:
        return float(x)


class IntegerAttribute(Attribute[int]):
    def is_valid(self, x: str) -> bool:
        try:
            self.parse(x)
            return True
        except ValueError:
            return False

    def parse(self, x: str) -> int:
        return int(x)


class NominalAttribute(Attribute[str]):
    def __init__(self, name: str, allowed_values: List[str]):
        super().__init__(name)
        self.allowed_values = allowed_values

    def is_valid(self, x: str) -> bool:
        return x.strip() in self.allowed_values

    def parse(self, x: str) -> T:
        return x


class DateAttribute(Attribute[datetime.datetime]):
    def __init__(self, name: str, isoformat: str):
        super().__init__(name)
        self.isoformat = isoformat

    def is_valid(self, x: str) -> bool:
        try:
            self.parse(x)
            return True
        except ValueError:
            return False

    def parse(self, x: str) -> datetime.datetime:
        return datetime.datetime.strptime(x, self.isoformat)


# endregion


def parse_arff_file(fname: str) -> DataFrame:
    with open(fname) as f:
        lines = f.readlines()

        readingHeader = True
        attributes: List[Attribute] = []  # Empty List

        data = []

        for line in lines:
            line = line.strip()

            if line.startswith("%"):  # Line is a comment
                continue

            if readingHeader:
                if line.lower().startswith("@relation"):  # Skip relation name
                    continue

                if line.lower().startswith("@data"):
                    readingHeader = False
                    continue

                if line.lower().startswith("@attribute"):  # Attribute Definition
                    parts = line.split(' ', 2)
                    attr_name = parts[1]
                    attr_type = parts[2]

                    if attr_type.lower() == "numeric" or attr_type.lower() == "real":
                        attributes.append(NumericAttribute(attr_name))
                    elif attr_type.lower() == "integer":
                        attributes.append(IntegerAttribute(attr_name))
                    elif attr_type.lower().startswith("date"):
                        date_parts = attr_type.split(' ', 2)
                        attributes.append(DateAttribute(attr_name, date_parts[1]))
                    else:
                        if not attr_type.startswith("{") or not attr_type.endswith("}"):
                            raise ValueError(f"Invalid Nominal Attribute Specification: {attr_type}")

                        values = []
                        for value in attr_type.split(','):
                            values.append(value.strip())

                        attributes.append(NominalAttribute(attr_name, values))
                    continue
            else:
                values = [x.strip() for x in line.split(',')]
                parsed = []

                if len(values) != len(attributes):
                    raise ValueError(f"Expected {len(attributes)} attributes in data line, but found {len(values)}")

                for i in range(len(values)):
                    if values[i] == '?':
                        parsed.append(None)
                    else:
                        parsed.append(attributes[i].parse(values[i]))

                data.append(parsed)
                continue

        col_names = [attr.name for attr in attributes]

        return DataFrame(data=data, columns=col_names)
