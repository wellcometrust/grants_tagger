import pandas as pd
import awswrangler as wr
import argparse
import random
from tqdm import tqdm


random.seed(42)


def create_grants_sample(
    s3_url: str,
    num_parquet_files_to_consider: int,
    num_samples_per_cat: int,
    output_file: str = None,
):
    parquet_files = wr.s3.list_objects(s3_url)
    random.shuffle(parquet_files)

    all_dfs = []

    for idx in tqdm(range(num_parquet_files_to_consider)):
        df = wr.s3.read_parquet(
            parquet_files[idx], columns=["title", "abstract", "for_first_level_name"]
        )

        all_dfs.append(df)

    df = pd.concat(all_dfs)

    # Do stratified sampling based on for_first_level_name column
    df_sample = df.groupby("for_first_level_name", group_keys=False).apply(
        lambda x: x.sample(min(len(x), num_samples_per_cat))
    )

    if output_file:
        df_sample.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-url", type=str)
    parser.add_argument("--num-parquet-files-to-consider", type=int, default=10)
    parser.add_argument("--num-samples-per-cat", type=int, default=10)
    parser.add_argument("--output-file", type=str, default="grants_sample.json")
    args = parser.parse_args()

    create_grants_sample(
        args.s3_url,
        args.num_parquet_files_to_consider,
        args.num_samples_per_cat,
        args.output_file,
    )
