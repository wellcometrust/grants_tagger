import xml.etree.ElementTree as ET

from tqdm import tqdm
import pandas as pd
import typer


def filter_mesh_tags(mesh_metadata_path, filtered_mesh_tags_path):
    mesh_tree = ET.parse(mesh_metadata_path)
    mesh_Df = pd.DataFrame(columns=["DescriptorName", "DescriptorUI", "TreeNumberList"])

    for mesh in tqdm(mesh_tree.getroot()):
        try:
            # TreeNumberList e.g. A11.118.637.555.567.550.500.100
            mesh_tree = mesh[-2][0].text
            # DescriptorUI e.g. M000616943
            mesh_code = mesh[0].text
            # DescriptorName e.g. Mucosal-Associated Invariant T Cells
            mesh_name = mesh[1][0].text
        except IndexError:
            # TODO: Add logger
            # print("ERROR", file=sys.stderr)
            pass
        if (
            mesh_tree.startswith("C")
            and not mesh_tree.startswith("C22")
            or mesh_tree.startswith("F03")
        ):
            mesh_Df = mesh_Df.append(
                {
                    "DescriptorName": mesh_name,
                    "DescriptorUI": mesh_code,
                    "TreeNumberList": mesh_tree,
                },
                ignore_index=True,
            )
    mesh_Df.to_csv(filtered_mesh_tags_path)
    return mesh_Df


if __name__ == "__main__":
    typer.run(filter_mesh_tags)
