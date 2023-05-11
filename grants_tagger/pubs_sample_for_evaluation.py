import awswrangler as wr
import json
from tqdm import tqdm
from collections import defaultdict
import random


def create_publications_sample(
    grants_year,
    cat_of_interest,
    num_samples_per_cat,
    num_parquet_files_to_consider,
    s3_dir,
    output_file,
):


    # list all relevant parquet files
    s3_url = f"{s3_dir}publications/year={grants_year}/"
    parquet_files = wr.s3.list_objects(s3_url)

    # now import dimensions categories
    cat_url = f"{s3_dir}categories/raw_categories.csv"
    df_cat = wr.s3.read_csv(cat_url)

    # make a dictionary of cat id to name
    cat_id2name = df_cat.set_index("category_id").to_dict("index")

    # let's make a dictionary with the category name as a key and as a value all related pubs (as a list)
    cat_samples_dict = defaultdict(list)

    for idx in tqdm(
        range(min(len(parquet_files), num_parquet_files_to_consider))
    ):  # in order to limit to the number of parquet files
        parquet_file = parquet_files[idx]

        df = wr.s3.read_parquet(
            parquet_file, columns=["title", "mesh_terms", "abstract", "categories"]
        )

        # one publication can have more than one category
        df_explode = df.explode("categories", ignore_index=True)
        bool_of_interest = df_explode["categories"].apply(
            lambda x: x["category_type"] in cat_of_interest
            if isinstance(x, dict)
            else False
        )

        # only consider the categories of interest
        df_filtered = df_explode[bool_of_interest].copy()

        # make the mesh_terms a list rather than a numpy array (in order to save as a .json later)
        df_filtered["mesh_terms"] = df_filtered["mesh_terms"].apply(lambda x: list(x))

        df_filtered["key"] = df_filtered["categories"].apply(
            lambda x: (cat_id2name[x["id"]]["name"], x["category_type"])
        )
        df_filtered["value"] = df_filtered[["title", "abstract", "mesh_terms"]].to_dict(
            "records"
        )

        # a dataframe with the keys and values we'd like to poor into our dictionary
        df_final = df_filtered[["key", "value"]].set_index("key")

        # add items to the dictionary
        for key, value in df_final.iterrows():
            cat_samples_dict[key].append(value.value)

    # now let's create a list of samples
    samples = []

    for category in cat_samples_dict:
        try:  # draw samples without replacement
            samples.extend(
                random.sample(cat_samples_dict[category], num_samples_per_cat)
            )
        except (
            ValueError
        ):  # if a category has fewer than num_samples_per_cat of examples, allow for replacement when drawing
            print(
                f"category {category} has less than {num_samples_per_cat} samples. For this category we allow for sampling with replacement but consider increasing the number of parquet files to consider."
            )
            samples.extend(
                random.choices(cat_samples_dict[category], k=num_samples_per_cat)
            )

    # save samples as a .json file
    with open(output_file, "w") as fout:
        json.dump(samples, fout)

if __name__ == "__main__":
    create_publications_sample(
        grants_year="2022",
        cat_of_interest=["for", "hrcs_rac"],
        num_samples_per_cat=10,
        num_parquet_files_to_consider=5,
        s3_dir="s3://datalabs-data/dimensions/",
        output_file="data/processed/evaluation/pubs_sample_for_evaluation.json",
    )

