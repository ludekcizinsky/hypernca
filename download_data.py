import os
import argparse
import gdown


def download_data(destination):

    os.makedirs(destination, exist_ok=True)

    # Texture Images
    # - Original
    path = os.path.join(destination, "images", "flickr+dtd_128")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        image_dataset_url = (
            ""
        )
        gdown.download(image_dataset_url, f"{destination}/images/image_dataset.zip", quiet=False)
        os.system(f"unzip {destination}/images/image_dataset.zip -d {destination}/images/")
        os.remove(f"{destination}/images/image_dataset.zip")
    else:
        print("Image dataset is already downloaded.")
        print("Remove the existing folder at images/flickr+dtd_128 to download again.\n")

    # - NCA generated 
    path = os.path.join(destination, "images", "nca_flickr+dtd_128")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        image_dataset_url = (
            ""
        )
        gdown.download(image_dataset_url, f"{destination}/images/nca_image_dataset.zip", quiet=False)
        os.system(f"unzip {destination}/images/nca_image_dataset.zip -d {destination}/images/")
        os.remove(f"{destination}/images/nca_image_dataset.zip")
    else:
        print("NCA-Generated image dataset is already downloaded.")
        print("Remove the existing folder at images/nca_flickr+dtd_128 to download again.\n")


    # Pre-trained NCA weights
    path = os.path.join(destination, "pretrained_nca", "Flickr+DTD_NCA") 
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        weight_dataset_url = (
            ""
        )
        gdown.download(weight_dataset_url, f"{destination}/pretrained_nca/nca_dataset.zip", quiet=False)
        os.system(f"unzip {destination}/pretrained_nca/nca_dataset.zip -d {destination}/pretrained_nca/")
        os.remove(f"{destination}/pretrained_nca/nca_dataset.zip")
    else:
        print("NCA weights dataset is already downloaded.")
        print("Remove the existing folder at pretrained_nca/Flickr+DTD_NCA to download again.\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Download and extract NCA dataset")
    parser.add_argument("--dest_path", type=str, help="Destination path to extract dataset", default="/scratch/izar/cizinsky/hypernca")
    args = parser.parse_args()
    download_data(args.dest_path)