#!/bin/bash
import pandas as pd

import os

from datascience.warehouse.warehouse import FortyTwo

forty_two = FortyTwo()
df = pd.DataFrame(forty_two.get_grants())

columns_of_interest = [
    'Grant ID',
    'Cost Centre Division Name',
    'Master Grant Type Name',
    'Reference',
    'Title',
    'Synopsis',
    'Lay Summary',
    'Research Question',
]

here = os.path.abspath(os.path.dirname(__file__))
df[columns_of_interest].to_csv(os.path.join(here, '../data/raw/grants.csv'))

print(f"Downloaded {len(df)} grants")
