from .constants import GCS_BUCKET_NAME, GCS_DATA_FOLDER
from .gcs_utils import upload_to_gcs, download_from_gcs

NUM_FILES = [1000, 10000]


def upload_all_train_data():
    for num_file in NUM_FILES:
        # find all file in data folder
        upload_to_gcs(GCS_BUCKET_NAME, f'../data/python_{num_file}.txt', f'{GCS_DATA_FOLDER}/python_{num_file}.txt')


def download_train_data(num_file):
    download_from_gcs(GCS_BUCKET_NAME, f'{GCS_DATA_FOLDER}/python_{num_file}.txt', f'../data/python_{num_file}.txt')
