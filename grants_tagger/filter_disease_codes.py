"""
Filters mesh tags to only disease specific (that start with C)
"""
from argparse import ArgumentParser
from pathlib import Path
import xml.etree.ElementTree as ET
import sys
import os

import pandas as pd

def filter_disease_codes(mesh_descriptions_file, mesh_export_file):
    mesh_tree = ET.parse(mesh_descriptions_file)
    mesh_Df = pd.DataFrame(columns = ['DescriptorName', 'TreeNumberList'])

    for mesh in mesh_tree.getroot():
        try:
            # TreeNumberList e.g. A11.118.637.555.567.550.500.100
            mesh_code = mesh[-2][0].text
            # DescriptorName e.g. Mucosal-Associated Invariant T Cells
            mesh_name = mesh[1][0].text
        except IndexError:
            print("ERROR", file=sys.stderr)
        if mesh_code.startswith('C') and not mesh_code.startswith('C22') or mesh_code.startswith('F03'):
            print(mesh_name)
            mesh_Df = mesh_Df.append({'DescriptorName':mesh_name, 'TreeNumberList':mesh_code}, ignore_index=True)
    mesh_Df.to_csv(mesh_export_file)

if __name__ == '__main__':
    argparser = ArgumentParser(description=__file__)
    argparser.add_argument('--mesh_descriptions_file', type=Path, help="path to xml file containing MeSH taxonomy")
    argparser.add_argument('--mesh_export_file', type=Path, help="path to csv file to export Mesh terms")
    argparser.add_argument('--config', type=Path, help="path to config file defining arguments")
    args = argparser.parse_args()

    if args.config:
        config_parser = ConfigParser()
        cfg = config_parser.read(args.config)

        mesh_descriptions_file = cfg["filter_disease_codes"]["mesh_descriptions_file"]
        mesh_export_file = cfg["filter_disease_codes"]["mesh_export_file"]
    else:
        mesh_descriptions_file = args.mesh_descriptions_file
        mesh_export_file = args.mesh_export_file

    if os.path.exists(mesh_export_file):
        print(f"{mesh_export_file} exists. Remove if you want to rerun")
    else:
        filter_disease_codes(mesh_descriptions_file, mesh_export_file)
