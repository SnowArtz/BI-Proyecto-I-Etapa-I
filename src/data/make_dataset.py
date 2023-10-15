# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from preprocessing import DataProcessor
from split_data import DataSplitter

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    preprocessor = DataProcessor(input_filepath)
    preprocessor.process_data()
    preprocessor.write_data("data/processed/cat_6716.csv")
    logger.info('Data set raw data saved')

    logger.info('Splitting data into train and test sets')
    dataSplitter = DataSplitter(input_filepath)
    dataSplitter.split_data_train_test()
    dataSplitter.write_data()
    logger.info('Data set split into train and test sets')

    logger.info('Processing splitted raw data')
    for file in ["X_train_raw.csv", "X_test_raw.csv"]:
        preprocessor = DataProcessor("data/interim/"+file)
        preprocessor.process_data()
        preprocessor.write_data("data/processed/"+file.replace("raw", "processed"))
    logger.info('Splitted raw data processed')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    project_dir = Path(__file__).resolve().parents[2]
    load_dotenv(find_dotenv())
    main()