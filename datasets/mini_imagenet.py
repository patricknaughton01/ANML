from __future__ import print_function

import hashlib
import os
from os.path import join
import tarfile

import numpy as np
import torch.utils.data as data
from PIL import Image
import requests

import torchvision.transforms as transforms
from .utils import check_integrity, list_dir, list_files


class MiniImageNet(data.Dataset):
    """`MiniImageNet  Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``mini-imagenet-gdrive`` exists.
        background (bool, optional): If True, creates dataset from the "background" set, otherwise
            creates from the "evaluation" set. This terminology is defined by the authors.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
    folder = 'mini-imagenet-gdrive'
    file_ids = {
        "train": "107FTosYIeBn5QbynR46YG91nHcJ70whs",
        "val": "1hSMUMj5IRpf-nQs1OwgiQLmGZCN0KDWl",
        "test": "1yKyKgxcnGMIAnA_6Vr2ilbpHMc9COg-v"
    }
    # download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
    tars_md5 = {
        "train": '62af9b3c839974dad2d474e6325795af',
        "val": 'ab02f050b0bf66823e7acb0c1ac1bc6b',
        "test": "318185fc3e3bf8bc57de887d9682c666"
    }
    tars_class_count = {
        "train": 64,
        "val": 16,
        "test": 20
    }

    def __init__(self, root, background=True,
                 transform=None, target_transform=None,
                 download=False, train=True, all=False):
        if not os.path.isdir(root):
            os.makedirs(root)
        self.root = join(os.path.expanduser(root), self.folder)
        if not os.path.isdir(self.root):
            os.mkdir(self.root)
        self.background = background
        self.transform = transform
        self.target_transform = target_transform
        self.images_cached = {}

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.target_folder = join(self.root, self._get_target_folder())


        #  Omniglot has this two-layer abstraction: alphabet --> character --> images
        #  MiniImagenet has only one-layer abstraction: class --> images
        #  Thus we can just fill in self._flat_character_images, a list of tuple (image_name, class_label)
        # self._alphabets = list_dir(self.target_folder)
        # self._characters = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
        #                         for a in self._alphabets], [])
        # self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
        #                           for idx, character in enumerate(self._characters)]
        # self._flat_character_images = sum(self._character_images, [])
        self._classes = []  # this would just mean classes
        self._flat_character_images = []
        for class_idx, class_name in enumerate(list_dir(self.target_folder)):
            self._classes.append(class_name)
            for img_name in list_files(os.path.join(self.target_folder, class_name), '.jpg'):
                self._flat_character_images.append((img_name, class_idx))
        self.data = [x[0] for x in self._flat_character_images]
        self.targets = [x[1] for x in self._flat_character_images]
        self.data2 = []
        self.targets2 = []
        self.new_flat = []
        for a in range(int(len(self.targets) / 20)):
            start = a * 20
            if train:
                for b in range(start, start + 15):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
                    self.new_flat.append(self._flat_character_images[b])
                    # print(self.targets[start+b])
            else:
                for b in range(start + 15, start + 20):  # (start + 15, start + 20):
                    self.data2.append(self.data[b])
                    self.targets2.append(self.targets[b])
                    self.new_flat.append(self._flat_character_images[b])

        if all:
            pass
        else:
            self._flat_character_images = self.new_flat
            self.targets = self.targets2
            print(self.targets[0:30])
            self.data = self.data2

        # this should probably be np.max(self.targets) + 1?
        print("Total classes = ", np.max(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name = self.data[index]
        character_class = self.targets[index]
        image_path = join(self.target_folder, self._classes[character_class], image_name)

        if image_path not in self.images_cached:

            image = Image.open(image_path, mode='r').convert('RGB')  # L
            image = image.resize((28, 28), resample=Image.LANCZOS)
            image = np.array(image, dtype=np.float32)
            normalize = transforms.Normalize(mean=[0.92206 * 256, 0.92206 * 256, 0.92206 * 256],
                                             std=[0.08426 * 256 * 256, 0.08426 * 256 * 256,
                                                  0.08426 * 256 * 256])  # adjust means and std of input data
            self.transform = transforms.Compose([transforms.ToTensor(), normalize])
            if self.transform is not None:
                image = self.transform(image)

            self.images_cached[image_path] = image
        else:
            image = self.images_cached[image_path]

        if self.target_transform:
            character_class = self.target_transform(character_class)

        return image, character_class

    def _cache_data(self):
        pass

    def _check_integrity(self) -> bool:
        """
        _check_integrity() verifies if the mini_imagenet dataset is properly downloaded and extracted
        :return: True if all three sets are downloaded and extracted
        """

        # https://stackoverflow.com/questions/42339876/error-unicodedecodeerror-utf-8-codec-cant-decode-byte-0xff-in-position-0-in
        def compute_md5(file_path: str) -> str:
            """
            compute_md5() computes the md5 value based on a file's content
            :param file_path: the path to the file
            :return: the md5() value based on the given file's data
            """
            if not os.path.isfile(file_path):
                return ""
            with open(file_path, 'rb') as f:
                contents = f.read()
            return hashlib.md5(contents).hexdigest()

        tar_filename = self._get_target_folder()
        downloaded = self.tars_md5[tar_filename] == compute_md5(os.path.join(self.root, f"{tar_filename}.tar"))
        for set_name in self.tars_md5:
            downloaded = downloaded and self._check_download(set_name)
        extracted = True
        for set_name in self.tars_md5:
            extracted = self._check_extraction(set_name)
        return downloaded and extracted

    def _check_download(self, set_name: str) -> bool:
        return os.path.isfile(os.path.join(self.root, f"{set_name}.tar"))

    def _check_extraction(self, set_name: str) -> bool:
        set_dir = os.path.join(self.root, set_name)
        if not os.path.isdir(set_dir):
            return False
        class_dir = [item for item in os.listdir(set_dir) if os.path.isdir(os.path.join(set_dir, item))]
        return len(class_dir) == self.tars_class_count[set_name]

    def download(self):
        """
        download() grabs the mini-imagenet dataset from the web
        # Using implementation from
        # 1.) https://github.com/nsadawi/Download-Large-File-From-Google-Drive-Using-Python
        # 2.) https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
        # 3.) Google Drive File ID: https://stackoverflow.com/questions/15057907/how-to-get-the-file-id-so-i-can-perform-a-download-of-a-file-from-google-drive-a
        """
        if self._check_integrity():
            print('miniImageNet files already downloaded and verified')
            return

        # Download the tar file
        for set_name in self.file_ids:
            if not self._check_download(set_name):
                download_path = os.path.join(self.root, f"{set_name}.tar")
                self._download_file_from_google_drive(file_id=self.file_ids[set_name], destination=download_path)
                print(f"Downloading to {download_path}")
            self._extract_tar_file(set_name)

    def _download_file_from_google_drive(self, file_id: str, destination: str):
        """
        _download_file_from_google_drive() downloads the file shared on Google drive to the specified path
        :param file_id: id of the shared file, grabbed from the Google Drive webpage
        :param destination: path on local disk to put the file
        :return: None
        """

        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768
            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        # magic url
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()
        response = session.get(URL, params={'id': file_id}, stream=True)
        token = get_confirm_token(response)
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        # save to disk
        save_response_content(response, destination)

    def _extract_tar_file(self, set_name: str):
        """
        _extract_tar_file() will extracts train.tar, val.tar, test.tar into their own folders
        Python Doc: https://docs.python.org/3.8/library/tarfile.html#tarfile.TarFile.extractall
        Stackoverflow: https://stackoverflow.com/questions/31163668/how-do-i-extract-a-tar-file-using-python-2-4
        :param set_name: a string denoting which split to extract
        :return: None
        """
        with tarfile.open(os.path.join(self.root, f"{set_name}.tar"), "r") as tar:
            # extracting to root = ../data/mini_imagenet/mini-imagenet-gdrive to avoid that extra layer
            tar.extractall(path=self.root)
        print(f"Extracting downloaded tar file to {self.root}")

    def _get_target_folder(self):
        return 'train' if self.background else 'val'
