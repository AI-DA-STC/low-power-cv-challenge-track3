import requests
import tarfile
import os
from tqdm import tqdm 

import logging
logger = logging.getLogger(__name__)

import logger as logging
logging.setup_logging()

def download(url, output_folder):


    filename = url.split("/")[-1]  # Extract filename from URL
    file_path = os.path.join(output_folder, filename)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get file size for progress bar
    response = requests.head(url)
    file_size = int(response.headers.get("Content-Length", 0))

    # Download the file with progress bar
    logger.info(f"Downloading {filename} ({file_size / (1024 * 1024):.2f} MB)...")
    with requests.get(url, stream=True) as response, open(file_path, "wb") as file, tqdm(
        total=file_size, unit="B", unit_scale=True, desc="Downloading"
    ) as progress_bar:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
            progress_bar.update(len(chunk))

    logger.info("\nDownload complete.")

    # Extract the tar file
    if file_path.endswith(".tar"):
        logger.info("Extracting files...")
        with tarfile.open(file_path, "r:") as tar:
            tar.extractall(path=output_folder)
        logger.info("Extraction complete.")

    return output_folder

if __name__ == "__main__":
    #download places365 dataset
    download("http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar", "./data/places365/")
    #download corresponding indoor outdoor labels for distribution analysis
    download("https://raw.githubusercontent.com/CSAILVision/places365/refs/heads/master/IO_places365.txt", "./data/places365/metadata/")