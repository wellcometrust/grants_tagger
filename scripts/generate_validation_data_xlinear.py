""" Script to generate validation data from Wellcome Grants """
import os

import pandas as pd

try:
    from datascience.warehouse.warehouse import FortyTwo
    from datascience.grants.cleaning import clean_grants
except ImportError as e:
    raise ImportError(f"Error importing {e.name }. "
                      f"To use this script you need to install Wellcome's datascience internal "
                      f"utils, for example with `make install-private-requirements`")


# Name of the file output of the pipeline
predictions_file = 'mesh_pipeline_result.csv'

# Name of the file with the predicitons, alongside metadata
merged_predictions_file = 'merged_mesh_predictions_mesh_xlinear_for_validation.csv'
here = os.path.abspath(os.path.dirname(__file__))

grants = pd.read_csv(os.path.join(here, '../data/raw/grants.csv'), index_col=0)
grants = clean_grants(grants).fillna('')

## Query FortyTwo to obtain the MeSH descriptors (could query the raw XML too?)
descriptor_query = """
SELECT [MeSH Lowest Level Classification], [MeSH Classification Description]
FROM FortyTwo.dbo.MeSHClassificationHierarchy 
"""

forty_two = FortyTwo()
descriptors = {
    row['MeSH Lowest Level Classification']: (
	row['MeSH Classification Description'].replace("\n", "") 
        if row['MeSH Classification Description'] else ""
    )
    for row in forty_two.execute_query(descriptor_query)
}

predictions = pd.read_csv(os.path.join(here, f'../data/interim/{predictions_file}'))
predictions.rename({'Grant id': 'Grant ID'}, axis=1, inplace=True)

merged_predictions_metadata = pd.merge(left=predictions, right=grants, how='left', on='Grant ID')
merged_predictions_metadata['MeSH Classification Description'] = merged_predictions_metadata['Tag'].apply(lambda x: descriptors.get(x, ""))
merged_predictions_metadata = \
    merged_predictions_metadata[merged_predictions_metadata['Start Date'] > '2012']
merged_predictions_metadata.to_csv(
    os.path.join(here, f'../data/processed/{merged_predictions_file}'),
    index=None
)
