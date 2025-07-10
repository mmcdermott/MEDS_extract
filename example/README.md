# Example Extraction Dataset

In this directory, we'll show an example of how to use MEDS-Extract on real (synthetic) patient data. Largely
speaking, this example will be run through a Jupyter Notebook included in `example.ipynb`. In this README,
we'll provide a brief overview of the process, the data, and how to run the example.

## Overview

This example will work with a simple raw dataset of patients, lab test results, diagnoses, and medications.
These files are stored in the [`raw_data`](raw_data) directory. The goal of this example is to show how to run
the MEDS-Extract pipeline end-to-end to obtain a MEDS dataset.

## Running the notebook

If Jupyter is installed, you can run the notebook from a locally cloned repository with
`jupyter notebook example.ipynb`. This will open a Jupyter Notebook in your browser, and you can run the cells
therein. If you wish to load it into Google Colab, you must also upload the `raw_data` directory to your local
Runtime or it will not find the raw files.

## Input Synthetic Data Generation Process

Our synthetic data generation process is simple: Ask ChatGPT (o3) to do it! We used the below prompt:

> I'd like you to generate a sample EHR dataset in the following format:
>
> 1. Generate at least 10 patients
> 2. There should be 4 files:
>     A) patients.csv, with the patient ID (as an integer, not a string), their eye color, their hair color their datetime of birth and datetime of death.
>     B) labs_vitals.csv, with a column for the lab test name, the patient ID, the timestamp, and the result (numerically). Include a handful of reasonable lab tests or vital signs that could be collected.
>     C) diagnoses.csv, with a column for the patient ID, the diagnosis code (ICD10), and the timestamp at which the diagnosis is given. Try to have the labs_vitals and diagnoses be a little consistent.
>     D) medications.csv, with a column for the medication name, the dose, the patient ID, and the timestamp.
> 3. For at least one patient, have more than 20 unique measurements for that patient.

This resulted in the files in the `raw_data` directory.

## Contributions

To install jupyter (and any other requirements for the example) alongside the other dependencies, you can run
`pip install -e .[example]` (potentially in addition to `dev` or `test` dependencies). The notebook for this
process will be run by default in the testing process, and thus should be validated before any pull requests
can be merged.
