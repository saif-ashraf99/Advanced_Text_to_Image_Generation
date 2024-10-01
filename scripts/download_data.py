import os
import requests
import zipfile

def download_and_extract(url, extract_to='.'):
    local_filename = url.split('/')[-1]
    print(f"Downloading {local_filename}...")
    response = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        f.write(response.content)
    print(f"Extracting {local_filename}...")
    with zipfile.ZipFile(local_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(local_filename)

if __name__ == '__main__':
    os.makedirs('data/raw/images', exist_ok=True)
    os.chdir('data/raw/images')
    download_and_extract('http://images.cocodataset.org/zips/train2017.zip')
    os.chdir('../../../')
    os.makedirs('data/raw/captions', exist_ok=True)
    os.chdir('data/raw/captions')
    download_and_extract('http://images.cocodataset.org/annotations/annotations_trainval2017.zip')
