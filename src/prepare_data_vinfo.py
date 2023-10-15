"""
This module is used for cleaning dataset files and transforming the
input to extract a particular attribute (e.g., the hypothesis-premise
overlap in SNLI).

Each dataset has a parent class in which the cleaning is done and several
subclasses, one for each transformation.
"""
import os
import pandas as pd
import logging
import argparse

parser = argparse.ArgumentParser()


# ------------------------------------------------------- DataTransformation -------------------------------------------------------

class DataTransformation(object):
    def __init__(self, transform_type, dataset, fname, output_dir):
        self.data = pd.read_csv(fname)
        self.data = self.data.rename({"text" : "sentence1"}, axis=1)
        self.dataset = dataset
        self.transform_type = transform_type
        self.output_dir = output_dir

    def transformation(self, example):
        raise NotImplementedError

    def transform(self):
        print(f'Applying {self.transform_type} to {self.dataset}\n')
        out_file = os.path.join(self.output_dir, f'{self.dataset}_{self.transform_type}.csv')
        print(f"Saving to {out_file}\n")
        self.data.apply(self.transformation, axis=1).to_csv(out_file, index=False)
        

class DataStandardTransformation(DataTransformation):
    def __init__(self, dataset, file_name, output_dir):
        super().__init__('std', dataset, file_name, output_dir)
        print("\nStd transformation...\n")

    def transformation(self, example):
        return example
         
         

class DataNullTransformation(DataTransformation):
    def __init__(self, dataset, file_name, output_dir):
        super().__init__('null', dataset, file_name, output_dir)
        print("\nNull transformation...\n")

    def transformation(self, example):
        example['sentence1'] = " " # using only empty string can yield problems
        return example



# ------------------------------------------------------- Main -------------------------------------------------------

if __name__ == "__main__":
    
    print(f"\nProcess ID: {os.getpid()}\n")

    parser.add_argument('--dataset', help='dataset name', required=True, type=str)
    parser.add_argument('--fname', help='file name', required=True, type=str)
    parser.add_argument('--output_dir', help='Output dataset dir', required=True, type=str)
    args = parser.parse_args()
    
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    DataStandardTransformation(args.dataset, args.fname, args.output_dir).transform()
    DataNullTransformation(args.dataset, args.fname, args.output_dir).transform()
    print("\nData computation done!\n")