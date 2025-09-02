import json
import numpy as np
import os
import re
import pandas as pd

"""README"""
# To run this file, please specify the input folder where you have the "browed.json"
# file extracted from discovery (ask Cohort Builder developer (Sepher)).
# This file must contains both layer thicknesses and CVIs from all the subjects
# you want to analyze. If something is missing from discovery, the analysis won't
# be working.
# Make sure to specify the out_path to be the base directories of the subjects folder
# and make sure the subject and session directories exist
# ALSo make sure you added the patient in left eye list if AOSLO images
# have been taken in the left eye (red in DATA_HC+HM.xlsx)
# WARNING: does not handle multiple session in subject path


# Opening JSON file
in_folder = r"C:\Users\BardetJ\Documents\layer_thickness"
# in_folder = r"T:\Studies\CohortBuilder\results\20230323 AOSLO"
out_path = r"C:\Users\BardetJ\Documents\mikhail\dataset\alain"
# patient_file = r"C:\Users\BardetJ\Documents\random\DATA_HC+DM.xlsx"
# patient_df = pd.read_excel(patient_file)
# patient_file = r"P:"
# out_path = r"P:AOSLO_"
f = open(os.path.join(in_folder, "browsed.json"))

lef_eye_subjects = [101, 28, 34]

# returns JSON object as
# a dictionary
data = json.load(f)

# get only the TVSS from the OCT cubes for each patients
for patient, studies in data.items():
    output_found = True
    patient_number = re.search(("\d+"), patient).group(0)
    i=0
    # Subject path
    complete_out_path = os.path.join(out_path, "Subject"+patient_number)
    # Session path
    session_path = [folder for folder in os.listdir(complete_out_path) if "Session" in folder][0]
    complete_out_path = os.path.join(complete_out_path, session_path)
    complete_out_path = os.path.join(complete_out_path, "layer")
    os.makedirs(complete_out_path, exist_ok=True)
    number_saved = 0
    for study, datasets in studies.items():
        for dataset in datasets:
            if 'OCT' not in dataset['info']['layerTypes']:
                continue
            if not 'OCT_CUBE' in dataset['info']['layerVariants']:
                continue
            if eval(patient_number) in lef_eye_subjects:
                if not dataset['info']['laterality'] == "L":
                    continue
            else:
                if not dataset['info']['laterality'] == "R":
                    continue
            # scale as space between each pixel scan (x), and space between
            # B-scans (y) -> (x,y)
            scale = dataset['oct']['info']['scale']
            out_data = {study: dataset['tvss'], "spacing": [scale[3], scale[0]]}
            json_object = json.dumps(out_data, indent=4)
            # print(f"save layers {complete_out_path}")
            # print(dataset['info'])
            with open(os.path.join(complete_out_path, "layer_thickness.json"), "w") as outfile:
                outfile.write(json_object)

            # also save cvis
            scale = dataset['oct']['info']['scale']
            out_data = {study: dataset['cvis'], "spacing": [scale[3], scale[0]]}
            json_object = json.dumps(out_data, indent=4)
            # print(f"save cvis {complete_out_path}")
            with open(os.path.join(complete_out_path, "layer_cvis.json"), "w") as outfile:
                outfile.write(json_object)
            number_saved+=1
    if number_saved>1:
        # why are there mutliple recordings dor one patient ???
        # maybe because they had to restart the OCT, so I will take the seecond (best?) one
        continue