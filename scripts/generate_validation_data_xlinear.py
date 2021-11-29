
grants = pd.read_csv('../data/raw/grants.csv', index_col=0)
grants = cleaning.clean_grants(grants).fillna('')

predictions = pd.read_csv('../data/processed/mesh_predictions_xlinear_{version}.csv')
predictions.rename({'Grant id': 'Grant ID'}, axis=1, inplace=True)

merged_predictions_metadata = pd.merge(left=predictions, right=grants, how='left', on='Grant ID')
merged_predictions_metadata = merged_predictions_metadata[merged_predictions_metadata['Start Date'] > '2012']
merged_predictions_metadata[::-1].to_csv('../data/processed/merged_predictions_mesh_2021.csv', index=None)
