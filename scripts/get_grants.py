
#!/bin/bash
import pandas as pd

import os
try:
    from datascience.warehouse.warehouse import FortyTwo
    from datascience.grants.cleaning import clean_grants
except ImportError as e:
    raise ImportError(f"Error importing {e.name }. "
                      f"To use this script you need to install Wellcome's datascience internal "
                      f"utils, for example with `make install-private-requirements`")

forty_two = FortyTwo()
df = pd.DataFrame(forty_two.get_grants())

columns_of_interest = [
    'Grant ID',
    'Reference',
    'Cost Centre Division Name',
    'Master Grant Type Name',
    'Funding Area',
    'Start Date',
    'End Date',
    'Title',
    'Is Awarded?',
    'Synopsis',
    'Lay Summary',
    'Research Question',
]

# Cleans grants to remove 'No data entered' and other things
df = clean_grants(df)

here = os.path.abspath(os.path.dirname(__file__))
df[columns_of_interest].to_csv(os.path.join(here, '../data/raw/grants.csv'))

print(f"Downloaded {len(df)} grants")
