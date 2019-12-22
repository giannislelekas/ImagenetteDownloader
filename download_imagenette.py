import json
import os
import glob
import pickle

import h5py
import wget
import tarfile
import numpy as np
import matplotlib.image as mpimg
from skimage.transform import resize


class ImagenetteLoader:

    def __init__(self, root_dir, image_format, new_shape=None, save_filepath=None):

        self.root_dir = root_dir
        self.image_format = image_format

        if save_filepath is None:
            self.save_filepath = root_dir

        if new_shape is not None:
            self.new_shape = new_shape
        else:
            self.new_shape = []

        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []
        self.classes_mapping = {}

    def download_files(self):

        if not os.path.exists(self.root_dir):
            print('Downloading Imagenette2')
            wget.download('https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz', './')

            print('Downloading complete')

            print('Extracting files')
            tar = tarfile.open('./imagenette2.tgz', "r:gz")
            tar.extractall()
            tar.close()
            print('Extracting complete')

            wget.download('https://raw.githubusercontent.com/ozendelait/wordnet-to-json/master/mapping_imagenet.json',
                          './imagenette2/')

    def load_json(self, filepath):

        with open(filepath, 'r') as f:
            return json.load(f)

    def get_image_paths(self, filepath):

        files = np.array(os.listdir(filepath))

        if not self.classes_mapping:
            mapping = self.load_json(self.root_dir + 'mapping_imagenet.json')
            c = 0
            for file in files:
                for _, j in enumerate(mapping):
                    if j['v3p0'] == file:
                        self.classes_mapping[c] = j['label'].split(',')[0]
                        c += 1
                    if c == len(files):
                        break

        paths = []
        labels = []
        c = 0
        for file in files:
            file = file.strip()
            image_paths = glob.glob(os.path.join(filepath, file, "*." + self.image_format))
            if image_paths:
                paths.extend(image_paths)
                labels.extend(c * np.ones(len(image_paths), dtype='uint8'))
            c += 1

        return files, paths, np.array(labels)

    def open_image(self, image_path):

        img = mpimg.imread(image_path)
        if self.new_shape:
            img = resize(img, (self.new_shape, self.new_shape), anti_aliasing=True)
        if len(img.shape) < 3:
            img = np.tile(img[..., np.newaxis], [1, 1, 3])
        return img.astype('uint8')

    def load_arrays(self):

        print('Loading Imagenette')

        _, paths, self.y_train = self.get_image_paths(self.root_dir + 'train/')
        for path in paths:
            self.x_train.append(self.open_image(path))

        _, paths, self.y_val = self.get_image_paths(self.root_dir + 'val/')
        for path in paths:
            self.x_val.append(self.open_image(path))

        self.x_train, self.x_val = np.array(self.x_train), np.array(self.x_val)

        print('Loading completed')

    def get_arrays(self):

        return self.x_train, self.y_train, self.x_val, self.y_val

    def save_set(self):

        def store(f, dset, name):

            c = 0
            for x in dset:
                f.create_dataset(np.string_(name + str(c)), data=x, compression='gzip', compression_opts=9)
                c += 1

        print(f'Storing Imagenette at {self.save_filepath}')

        f = h5py.File(self.save_filepath + 'imagenette.h5', 'w')
        store(f, self.x_train, 'x_train/')
        f.create_dataset('y_train', (len(self.y_train),), data=self.y_train, dtype=self.y_train.dtype)
        store(f, self.x_val, 'x_val/')
        f.create_dataset('y_val', (len(self.y_val),), data=self.y_val, dtype=self.y_val.dtype)

        with open(self.save_filepath + 'classes_mapping.pkl', 'wb') as d:
            pickle.dump(self.classes_mapping, d)

        f.close()

        print('Storing completed')

    def load(self):

        self.download_files()
        self.load_arrays()
        self.save_set()

def load_imagenette(filepath):

    f = h5py.File(filepath + 'imagenette.h5', 'r')
    data = {}

    groups = list(f.keys())
    for group in groups:
        data[group] = []

    for group in groups:
        if isinstance(f[group], h5py.Dataset):
            data[group].extend(f[group][:])
        else:
            subgroups = list(f[group].keys())
            for c in range(len(subgroups)):
                data[group].append(f[group + '/' + str(c)][:])
        data[group] = np.array(data[group])

    with open(filepath + 'classes_mapping.pkl', 'rb') as h:
        classes_mapping = pickle.load(h)

    f.close()
    h.close()

    return data, classes_mapping


def main():
    loader = ImagenetteLoader('./imagenette2/', 'JPEG', new_shape=256)
    loader.load()

    # d = load_imagenette('./imagenette2/imagenette.h5')


if __name__ == '__main__':
    main()
