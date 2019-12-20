import os
import glob
import wget
import tarfile
import numpy as np
import matplotlib.image as mpimg

class ImagenetteLoader:

    def __init__(self, root_dir, image_format, download=False, save_filepath=None):

        self.root_dir = root_dir
        self.image_format = image_format
        self.download = download

        if save_filepath is None:
            self.save_filepath = root_dir

        self.x_train = []
        self.y_train = []
        self.x_val = []
        self.y_val = []

    def download_files(self):

        if self.download:
            print('Downloading Imagenette2')
            wget.download('https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz', './')
            print('Downloading complete')

            print('Extracting files')
            tar = tarfile.open('./imagenette2.tgz', "r:gz")
            tar.extractall()
            tar.close()
            print('Extracting complete')

    def get_image_paths(self, filepath):

        files = np.array(os.listdir(filepath))
        ind = np.argsort([(int(file[1:])) for file in files])
        files = files[ind]
        paths = []
        labels = []
        c = 0
        for file in files:
            file = file.strip()
            image_paths = glob.glob(os.path.join(filepath, file, "*." + self.image_format))
            if image_paths:
                paths.extend(image_paths)
                labels.append(c * np.ones(len(image_paths)))
            c += 1
        labels = np.array([l for sublist in labels for l in sublist])

        return files, paths, labels

    def open_image(self, image_path):

        img = mpimg.imread(image_path)
        return img.astype('uint8')

    def load_arrays(self):

        print('Loading Imagenette')
        x_train = []
        x_val = []

        _, paths, self.y_train = self.get_image_paths(self.root_dir + 'train/')
        for path in paths:
            x_train.append(self.open_image(path))

        _, paths, self.y_val = self.get_image_paths(self.root_dir + 'val/')
        for path in paths:
            x_val.append(self.open_image(path))

        self.x_train, self.x_val = np.array(x_train), np.array(x_val)

        print('Loading completed')

    def get_arrays(self):

        return self.x_train, self.y_train, self.x_val, self.y_val

    def save_set(self):

        print(f'Storing Imagenette at {self.save_filepath}')
        np.savez(self.save_filepath + 'imagenette.npz', self.get_arrays())
        print('Storing completed')

    def load(self):

        self.download_files()
        self.load_arrays()
        self.save_set()

def main():

    loader = ImagenetteLoader('./imagenette2/', 'JPEG', download=False)
    loader.load()


if __name__ == '__main__':
    main()
