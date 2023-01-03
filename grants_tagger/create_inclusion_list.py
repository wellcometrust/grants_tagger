"""
Script to automatically create a list of MeSH terms to include
(i.e. filtering out the ones that have an attribute which mentions "Do not use")
given a link to the MeSH xml file
"""
import typer

import os
import uuid
import xml.etree.ElementTree as ET  # this is a library that can parse xml easily
import pandas as pd
from pathlib import Path


def inclusion(xml_path, exclusion_list_path, out_path):
    """
    Creates an inclusion list based on a manual exclusion list and a filter
    to filter out terms which have an attribute mentioning "Do not use"

      Args:
      - xml_path: path to the xml which contains the MeSH tree and term attributes
      - exclusion_list_path: path to .csv with terms we manually want to exclude
      - out_path: path to .csv which contains list of MeSH terms to use

      Returns:
        csv in the out_path
    """

    # parse the xml tree
    mesh_tree = ET.parse(xml_path)

    # get the annotations and descriptors
    annotations = []
    descriptors = []
    for mesh in mesh_tree.iter("DescriptorRecord"):
        descriptors.append(mesh.find("DescriptorName").find("String").text)
        annotation = mesh.find("Annotation")
        if annotation is None:
            annotations.append("")
        else:
            annotations.append(mesh.find("Annotation").text)

    descriptors_to_use = []
    descriptors_set = zip(descriptors, annotations)

    # read in the list of descriptors we would like to drop manually:
    exclusion_df = pd.read_csv(exclusion_list_path)
    exclusion_list = exclusion_df["DescriptorName"].tolist()

    # make a list of the descriptors we want to use
    for descriptor, annotation in descriptors_set:
        if (
            ": Do not use" not in annotation and descriptor not in exclusion_list
        ):  # by excluding these
            descriptors_to_use.append(descriptor)

    pd.DataFrame(descriptors_to_use, columns=["DescriptorName"]).to_csv(out_path)


# the below is to have a nice CLI interface just like in the other grants_tagger scripts
create_inclusion_app = typer.Typer()


@create_inclusion_app.command()
def inclusion_cli(
    mesh_xml_path: Path = typer.Argument(..., help="path to mesh xml"),
    exclusion_list_path: Path = typer.Argument(
        ..., help="path to manual exclusion list"
    ),
    out_path: Path = typer.Argument(..., help="path to output argument"),
):

    inclusion(mesh_xml_path, exclusion_list_path, out_path)


if __name__ == "__main__":
    create_inclusion_app()
