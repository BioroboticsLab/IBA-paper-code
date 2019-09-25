# from mlproject.data import *
import h5py
import io
import torch
import torchvision
import numpy as np
from os import path
import yaml
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import os
import copy
from torch.utils.data import Dataset, DataLoader
import errno
import pickle


class DatasetFactory:
    """
    The DatasetFactory provides access to the train/test/val datasets and all the dataset
    iterators.

    The `Dataset`'s returned by `train_set`, `test_set`, and `validation_set`
    should be persistent, e.g.  an index should always return the same data and
    label.

    The `DataLoaders` returned by *_loader can of course return the data
    augmented and in any order.
    """
    def __init__(self,
                 train_set=None, train_loader=None,
                 test_set=None, test_loader=None,
                 validation_set=None, validation_loader=None):
        self._train_set = train_set
        self._train_loader = train_loader
        self._test_set = test_set
        self._test_loader = test_loader
        self._validation_set = validation_set
        self._validation_loader = validation_loader

    def train_set(self) -> Dataset:
        """Return the train set."""
        return self._train_set

    def train_loader(self) -> DataLoader:
        """Return the DataLoader associated with the train set."""
        return self._train_loader

    def test_set(self) -> Dataset:
        """Return the test set."""
        return self._test_set

    def test_loader(self) -> DataLoader:
        """Return the DataLoader associated with the test set."""
        return self._test_loader

    def validation_set(self) -> Dataset:
        """Return the validation set."""
        return self._validation_set

    def validation_loader(self) -> DataLoader:
        """Return the DataLoader associated with the validation set."""
        return self._validation_loader

    def datasets(self):
        return self.train_set(), self.test_set(), self.validation_set()

    def dataloaders(self):
        return self.train_loader(), self.test_loader(), self.validation_loader()

    def has_train_set(self) -> bool:
        return self.train_set() is not None

    def has_test_set(self) -> bool:
        return self.test_set() is not None

    def has_validation_set(self) -> bool:
        return self.validation_set() is not None


class CycleDataLoader(DataLoader):
    def __init__(self, dataloader, n_cycles=1000):
        self.dataloader = dataloader
        self.n_cycles = n_cycles

    def __iter__(self):
        for _ in range(self.n_cycles):
            for batch in self.dataloader:
                yield batch

    def __len__(self):
        return len(self.dataloader) * self.n_cycles


class CycleDatasetFactory(DatasetFactory):
    def __init__(self, factory: DatasetFactory, n_cycles=1000):
        def cycle(loader):
            if loader is None:
                return None
            return CycleDataLoader(loader, n_cycles)
        super().__init__(
            factory.train_set(),
            cycle(factory.train_loader()),
            factory.test_set(),
            cycle(factory.test_loader()),
            factory.validation_set(),
            cycle(factory.validation_loader()),
        )


def default_data_dir(maybe_data_dir=None):
    if maybe_data_dir is not None:
        return maybe_data_dir
    elif "DATA_DIR" in os.environ:
        return os.environ['DATA_DIR']
    else:
        raise ValueError("Can not figure out data_dir. "
                         "Please set the DATA_DIR enviroment variable.")


class TorchvisionDatasetFactory(DatasetFactory):
    # TODO: Extract more code from subclass
    def __init__(self, train_set=None, test_set=None, validation_set=None,
                 data_loader_kwargs={},
                 data_loader_train_kwargs={},
                 data_loader_test_kwargs={}):
        train_kwargs = copy.copy(data_loader_kwargs)
        train_kwargs.update(data_loader_train_kwargs)
        test_kwargs = copy.copy(data_loader_kwargs)
        test_kwargs.update(data_loader_test_kwargs)

        if train_set is not None:
            trainloader = torch.utils.data.DataLoader(train_set, **train_kwargs)
        else:
            trainloader = None

        if test_set is not None:
            testloader = torch.utils.data.DataLoader(test_set, **test_kwargs)
        else:
            testloader = None

        if validation_set is not None:
            valloader = torch.utils.data.DataLoader(validation_set, **test_kwargs)
        else:
            valloader = None

        super().__init__(
            train_set, trainloader,
            test_set, testloader,
            validation_set, valloader,
        )


class CelebALabelList:
    def __init__(self, filenames, names, labels):
        self.filenames = filenames
        self.names = names
        self.labels = labels

    @staticmethod
    def load(filename):
        filename_npz = filename + '.npz'
        if os.path.exists(filename_npz):
            with open(filename_npz, 'rb') as f:
                npz = np.load(f)
                filenames = npz['filenames'].tolist()
                names = npz['label_names'].tolist()
                labels = npz['labels']
        else:
            filenames, names, labels = CelebALabelList._load_list(filename)
            with open(filename_npz, 'wb') as f:
                np.savez(f, filenames=filenames,
                         label_names=names,
                         labels=labels)
        return CelebALabelList(filenames, names, labels)

    @staticmethod
    def _load_list(filename):
        def split(line):
            collapsed_whitespace = ' '.join(line.rstrip(' \n').split())
            return collapsed_whitespace.split(' ')
        with open(filename) as f:
            lines = f.readlines()
            print(lines[1])
            attribute_names = split(lines[1])
            filenames = []
            labels = []
            for line in lines[2:]:
                values = split(line)
                filenames.append(values[0])
                labels.append(list(map(int, values[1:])))
        return filenames, attribute_names, np.array(labels, np.int32)

    def __getitem__(self, idx):
        return self.filenames[idx], self.labels[idx]

    def __len__(self):
        return len(self.filenames)

    def filter(self, func):
        mask = func(self.labels)
        return np.nonzero(mask)[0]


class CelebA(Dataset):
    def __init__(self, root_dir, aligned=True, partition='train', transform=ToTensor(),
                 labels='attributes'):
        """
        CelebA Dataset

        Args:
            root_dir (str): root directory
            aligned (bool): whether to use the aligned version (default: True)
            partition (str): use the ``train``, ``test`` or ``validation`` partion.
            transform: image transformation to apply
            labels (str or list): labels to return (default: attributes).
                Can be ``['bbox', 'landmarks', 'attributes', 'identity']``.
        """
        self.aligned = aligned
        self.transform = transform
        self.partition = partition
        if self.aligned:
            self.image_dir = os.path.join(root_dir, 'img_align_celeba_png')
        else:
            self.image_dir = os.path.join(root_dir, 'img_celeba')
        # identity_CelebA.txt  list_attr_celeba.txt  list_bbox_celeba.txt
        # list_landmarks_align_celeba.txt  list_landmarks_celeba.txt
        self.identity_file = os.path.join(root_dir, 'Anno/identity_CelebA.txt')
        self.identity = CelebALabelList.load(self.identity_file)
        self.attr_file = os.path.join(root_dir, 'Anno/list_attr_celeba.txt')
        self.attributes = CelebALabelList.load(self.attr_file)
        self.bbox_file = os.path.join(root_dir, 'Anno/list_bbox_celeba.txt')
        self.bbox = CelebALabelList.load(self.bbox_file)
        self.landmarks_align_file = os.path.join(root_dir, 'Anno/list_landmarks_align_celeba.txt')
        self.landmarks_file = os.path.join(root_dir, 'Anno/list_landmarks_celeba.txt')
        if self.aligned:
            self.landmarks = CelebALabelList.load(self.landmarks_align_file)
        else:
            self.landmarks = CelebALabelList.load(self.landmarks_file)
        self.eval_file = os.path.join(root_dir, 'Eval/list_eval_partition.txt')
        self.eval_partition = CelebALabelList.load(self.eval_file)
        self.labels = labels
        self.reset_selection()

    def _filter(self, func, label_list):
        mask = func(label_list.labels[self.selection])
        self.selection = self.selection[mask]

    def reset_selection(self):
        label_id = {'train': 0, 'validation': 1, 'test': 2}[self.partition]
        self.selection = self.eval_partition.filter(lambda x: x[:, 0] == label_id)

    def filter_by_identity(self, func):
        """``func`` recieves the labels and returns a binary mask """
        self._filter(func, self.identity)

    def filter_by_landmarks(self, func):
        """``func`` recieves the labels and returns a binary mask """
        self._filter(func, self.landmarks)

    def filter_by_attributes(self, func):
        """``func`` recieves the labels and returns a binary mask """
        self._filter(func, self.attributes)

    def filter_by_bbox(self, func):
        """``func`` recieves the labels and returns a binary mask """
        self._filter(func, self.bbox)

    def __getitem__(self, idx):
        real_idx = self.selection[idx]
        filename, _ = self.eval_partition[real_idx]
        if self.aligned:
            filename = filename[:-3] + 'png'
        img = Image.open(os.path.join(self.image_dir, filename))
        if type(self.labels) == list:
            labels = []
            for label in self.labels:
                label_list = getattr(self, label)
                labels.append(label_list[real_idx])
        else:
            label_list = getattr(self, self.labels)
            labels = label_list[real_idx]
        return self.transform(img), labels

    def __len__(self):
        return len(self.selection)


class CelebAFactory(TorchvisionDatasetFactory):
    def __init__(self, root, aligned=True, labels='attributes',
                 train_transform=ToTensor(), test_transform=ToTensor(),
                 data_loader_kwargs={},
                 data_loader_train_kwargs={},
                 data_loader_test_kwargs={}):
        train_set = CelebA(root, aligned, 'train', train_transform, labels)
        test_set = CelebA(root, aligned, 'test', test_transform, labels)
        validation_set = CelebA(root, aligned, 'validation', test_transform, labels)
        super().__init__(train_set, test_set, validation_set,
                         data_loader_kwargs, data_loader_train_kwargs,
                         data_loader_test_kwargs)


class ClutteredMNIST(Dataset):
    def __init__(self, dataset, shape=(100, 100), n_clutters=6, clutter_size=8,
                 n_samples=60000, transform=None):
        self.dataset = dataset
        self.shape = shape
        self.n_clutters = n_clutters
        self.clutter_size = clutter_size
        self.n_samples = n_samples
        self.transform = transform
        self._parameters = None  # are set on self._init() to save time when they are not needed

    def get_parameters(self):
        if self._parameters is None:
            self._init_parameters()
        return self._parameters

    def _init_parameters(self):
        self._parameters = self.generate_parameters()

    def export(self, datadir, force_write=False) -> bool:
        """
        write image files in the file system, which can be loaded with:
        https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
        or do nothing if they already exist
        structure: [datadir]/[class idx]/[sample idx].png
        :param force_write: if True, the files are written, even if they already exist
        :param datadir: the location for the data
        :return: boolean, True if the data was written, False if the data was already there
        """
        # TODO check if files are complete
        # TODO remove MNIST data after export completion
        # TODO on overwriting, remove all existing files first (prevent leftovers from old dataset)
        abspath = path.abspath(datadir)
        meta = {
            "n_samples": self.n_samples,
            "clutter_size": self.clutter_size,
            "n_clutters": self.n_clutters,
        }
        metapath = path.join(abspath, 'meta.p')
        if path.exists(metapath) and not force_write:
            # this seems to be a existing dataset folder. if it is identical, we can reuse it
            with open(metapath, mode='rb') as fp:
                existing_meta = pickle.load(fp)
                # check if meta is identical
                for name, item in meta.items():
                    if name not in existing_meta:
                        raise RuntimeError("There is already a dataset stored in this location "
                                           "and it does not specify the config value {}".format(name))
                    if existing_meta[name] != meta[name]:
                        raise RuntimeError("There is already a dataset stored in this location "
                                           "and the config value {} differs: {} != {}. Please "
                                           "remove the files or choose different location."
                                           "".format(name, existing_meta[name], meta[name]))
            # the files exist and are identical. we can quit.
            return False

        # the files do not yet exist. we will write them
        print("ClutteredMNIST: Exporting {} files to {}".format(self.n_samples, abspath))
        # write image files
        tmp_transform = self.transform  # store transform
        self.transform = None
        counters = list()
        for i, (pil_img, label) in enumerate(self):
            if isinstance(label, torch.Tensor):
                label = label.detach().cpu().numpy()
            while len(counters) <= label:
                counters.append(0)
            counters[label] += 1
            labelpath = self.ensure_dir(path.join(abspath, str(label)))
            pil_img.save(path.join(labelpath, str(counters[label])+".png"), format="png")
        self.transform = tmp_transform  # restore transform
        # write meta
        # add params to meta (we did not want to generate them earlier, in case we didn't need them)
        meta["params"] = self.get_parameters()
        with open(metapath, 'wb') as fp:
            pickle.dump(meta, fp, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def ensure_dir(dirname) -> str:
        """ check if dir exists, otherwise create it safely or raise error"""
        try:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        return dirname

    def generate_parameters(self):
        all_params = []
        h, w = self.dataset[0][0].size
        for i in range(self.n_samples):
            params = {
                'idx': i % len(self.dataset),
                'digit_h': np.random.randint(0, self.shape[0] - h),
                'digit_w': np.random.randint(0, self.shape[1] - w),
            }
            clutter = []
            for _ in range(self.n_clutters):
                clutter_idx = np.random.randint(0, len(self.dataset))
                cs = self.clutter_size
                ph = np.random.randint(0, h - cs)
                pw = np.random.randint(0, w - cs)
                ch = np.random.randint(0, self.shape[0] - cs)
                cw = np.random.randint(0, self.shape[1] - cs)
                clutter.append({
                    'clutter_idx': clutter_idx,
                    'patch_h': ph,
                    'patch_w': pw,
                    'clutter_h': ch,
                    'clutter_w': cw,
                })
            params['clutter'] = clutter
            all_params.append(params)
        return all_params

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self._parameters is None:
            self._init_parameters()
        canvas = np.zeros(self.shape, dtype=np.uint8)
        params = self._parameters[idx]
        for clutter in params['clutter']:
            clutter_img = np.array(self.dataset[clutter['clutter_idx']][0])
            h, w = clutter_img.shape
            # select patch
            cs = self.clutter_size
            ph = clutter['patch_h']
            pw = clutter['patch_w']
            patch = clutter_img[ph:ph+cs, pw:pw+cs]
            # place patch
            ch = clutter['clutter_h']
            cw = clutter['clutter_w']
            canvas[ch:ch+cs, cw:cw+cs] = patch

        img, label = self.dataset[params['idx']]
        img = np.array(img)
        h, w = img.shape
        dh = params['digit_h']
        dw = params['digit_w']
        canvas[dh:dh+h, dw:dw+w] = img
        pil_img = Image.fromarray(canvas, mode='L')
        if self.transform is not None:
            return self.transform(pil_img), label
        else:
            return pil_img, label



class CIFARDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True,
                                                download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False,
                                               download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )


class FashionMNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=1, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0, collate_fn=None, pin_memory=None, drop_last=None,
                 data_loader_train_kwargs=None,
                 data_loader_test_kwargs=None,
                 ):
        def update_kwargs(kwargs):
            kwargs = copy.copy(kwargs or {})
            if 'num_workers' not in kwargs:
                kwargs['num_workers'] = num_workers
            if 'collate_fn' not in kwargs and collate_fn is not None:
                kwargs['collate_fn'] = collate_fn
            if 'pin_memory' not in kwargs and pin_memory is not None:
                kwargs['pin_memory'] = pin_memory
            if 'drop_last' not in kwargs and drop_last is not None:
                kwargs['drop_last'] = drop_last
            if 'batch_size' not in kwargs:
                kwargs['batch_size'] = batch_size
            return kwargs

        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.FashionMNIST(root=self.data_dir, train=True,
                                                     download=True, transform=train_transform)
        testset = torchvision.datasets.FashionMNIST(root=self.data_dir, train=False,
                                                    download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_train_kwargs=update_kwargs(data_loader_train_kwargs),
            data_loader_test_kwargs=update_kwargs(data_loader_test_kwargs)
        )


class MNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 num_workers=0):
        self.data_dir = default_data_dir(data_dir)
        trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True,
                                              download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root=self.data_dir, train=False,
                                             download=True, transform=test_transform)
        super().__init__(
            trainset, testset,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )


class ClutteredMNISTDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size=50, train_transform=None, test_transform=None, data_dir=None,
                 shape=(100, 100), n_clutters=6, clutter_size=8, n_samples_train=60000, n_samples_test=10000, n_samples_val=10000,
                 num_workers=0, use_filesys=False):
        self.data_dir = default_data_dir(data_dir)
        # create dataset generators
        trainset = torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
        testset = torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)
        generator_train = ClutteredMNIST(trainset, shape, n_clutters,
                                         clutter_size, n_samples_train,
                                         transform=train_transform)
        generator_test = ClutteredMNIST(testset, shape, n_clutters,
                                        clutter_size, n_samples_test,
                                        transform=test_transform)
        if use_filesys:
            # export dataset files to data dir
            train_dir = path.join(data_dir, "train")
            test_dir = path.join(data_dir, "test")
            generator_train.export(train_dir)
            generator_test.export(test_dir)

            # create Torchvision ImageFolder Dataset of these dirs
            def loader(path) -> Image:
                return Image.open(path).convert("L")  # force one channel, just like original MNIST
            resource_train = torchvision.datasets.ImageFolder(train_dir, transform=train_transform, loader=loader)
            resource_test = torchvision.datasets.ImageFolder(test_dir, transform=test_transform, loader=loader)
        else:
            # use just-in-time generation
            resource_train = generator_train
            resource_test = generator_test

        # construct parent
        super().__init__(
            resource_train, resource_test,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )


class ImgAug:
    def __init__(self, transforms):
        from imgaug import augmenters as iaa
        self.transform = iaa.Sequential(transforms)

    def __call__(self, img):
        img = np.array(img)
        img = self.transform.augment_image(img)
        if len(img.shape) == 3 and img.shape[-1] == 3:
            mode = 'RGB'
        else:
            mode = 'L'
        return Image.fromarray(img, mode=mode)



class DataProvider:
    """
    wrapper class for a dataset factory to retrieve data from
    """
    def __init__(self, data_fac, device, label_dict=None):
        self.data_fac = data_fac
        self.device = device
        self.label_dict = label_dict

    def gen_samples(self, num, device=None, test_set=False) -> list:
        device = device if device else self.device
        loader = self.data_fac.test_loader() if test_set else self.data_fac.train_loader()
        yielded = 0
        for i, (data, labels) in enumerate(loader.__iter__()):
            done = False
            for s in range(data[0].shape[0]):
                yielded += 1
                if yielded <= num:
                    yield (data[s].unsqueeze(0).to(device), labels[s].unsqueeze(0).to(device))
                else:
                    done = True
                    break
            if done:
                break

    def gen_batches(self, num, device=None, test_set=False):
        device = device if device else self.device
        loader = self.data_fac.train_loader() if not test_set else self.data_fac.test_loader()
        for i, (data, labels) in enumerate(loader.__iter__()):
            if i > num:
                break
            yield (data.to(device), labels.to(device))

    def get_label(self, num):
        """ if a label dictionary was provided: read it, otherwise just return plain number """
        return str(num) if self.label_dict is None else self.label_dict[num]


class DataSubsetIterator:
    def __init__(self, loader, ):
        pass

    def __next__(self):
        pass


class TorchZooImageNetFolderDataProvider(DataProvider):
    """
    seed 1, get 100 samples:
        1=boar
        2=monkey (382)
    """
    def __init__(self, config, transform=None):
        # prepare arguments for ImageFolderDataset
        if transform is None:
            transform = Compose([
                Resize(256),
                CenterCrop((224, 224)),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.transform = transform

        # maybe seed
        seed = config.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(config["seed"])

        # maybe load label dict
        label_dict = None
        dict_file = config.get("imagenet_dict_file", None)
        if dict_file is not None:
            with open(dict_file, encoding='utf-8') as handle:
                label_dict = yaml.load(handle.read())

        device = config["device"]
        data_fac = ImageFolderDatasetFactory(
            batch_size=config["batch_size"],
            num_workers=10,
            data_train_dir=config.get("imagenet_train"),
            data_test_dir=config.get("imagenet_test"),
            train_transform=self.transform,
            test_transform=self.transform)

        super().__init__(data_fac, device, label_dict)

    def image_to_batch(self, path, label:int):
        sample = Image.open(path)
        sample = sample.convert('RGB')
        sample = (self.transform(sample).unsqueeze(0).to(self.device), torch.tensor(label).to(self.device))
        return sample


# dataset factories


class ImageNetDataset:
    def __init__(self, hdf5_filename, train, transform=None):
        self.hdf5_filename = hdf5_filename
        self.train = train
        self.dataset_name = 'train' if train else 'validation'
        self.transform = transform
        self.open = False
        self.h5 = None
        self.h5_images = None
        self.h5_targets = None

        with h5py.File(hdf5_filename, 'r') as tmp_h5:
            h5_targets = tmp_h5[self.dataset_name + '/targets']
            self.length = len(h5_targets)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not self.open:
            self.h5 = h5py.File(self.hdf5_filename, 'r', swmr=True)
            self.h5_images = self.h5[self.dataset_name + '/images']
            self.h5_targets = self.h5[self.dataset_name + '/targets']
            self.open = True
        target = self.h5_targets[idx]
        jpg_bytes = self.h5_images[idx].tobytes()
        pil_image = Image.open(io.BytesIO(jpg_bytes))
        if self.transform is not None:
            img = self.transform(pil_image)
        else:
            img = pil_image
        return img, int(target)


class ImageFolderDatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size,
                 data_train_dir,
                 data_test_dir,
                 train_transform=None,
                 test_transform=None,
                 loader=None,
                 num_workers=0):
        assert path.exists(data_train_dir), "train does not exist: {}".format(data_train_dir)
        assert path.exists(data_test_dir), "test does not exist: {}".format(data_test_dir)
        loader = loader if loader is not None else torchvision.datasets.folder.default_loader
        resource_train = torchvision.datasets.ImageFolder(data_train_dir, transform=train_transform, loader=loader)
        resource_test = torchvision.datasets.ImageFolder(data_test_dir, transform=test_transform, loader=loader)
        super().__init__(
            resource_train, resource_test,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
            },
            data_loader_train_kwargs={'shuffle': True}
        )


class ImageNetHDF5DatasetFactory(TorchvisionDatasetFactory):
    def __init__(self, batch_size,
                 train_transform=None,
                 test_transform=None,
                 data_file=None,
                 dict_file=None,
                 num_workers=0):
        resource_train = ImageNetDataset(data_file, transform=train_transform, train=True)
        resource_test = ImageNetDataset(data_file, transform=test_transform, train=False)
        with open(dict_file, encoding='utf-8') as handle:
            self.dict = yaml.load(handle.read())
        super().__init__(
            resource_train, resource_test,
            data_loader_kwargs={
                'num_workers': num_workers,
                'batch_size': batch_size,
                'shuffle': True,
            },
            data_loader_train_kwargs={}
        )

    def get_label(self, num):
        return self.dict[num]


class TorchZooHDF5ImageNetDatasetFactory(ImageNetHDF5DatasetFactory):
    def __init__(self, config):
        train_transform = Compose([
            Resize(256),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_transform = Compose([
            Resize(256),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        seed = config.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed()
        super().__init__(
            batch_size=config["batch_size"],
            num_workers=10,
            data_file=config.get("data_file"),
            dict_file=config.get("dict_file"),
            train_transform=train_transform,
            test_transform=test_transform)


class ClutMNISTNormalizedDataProvider(DataProvider):
    def __init__(self, config):

        # maybe seed
        seed = config.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed()

        config = self.apply_defaults(config)
        device = config["device"]
        data_resize = config.get("data_resize", (96, 96))
        train_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5,), (0.5,))])
        test_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5,), (0.5,))])
        data_fac = ClutteredMNISTDatasetFactory(
            batch_size=config.get("batch_size", 32),
            shape=(96, 96),
            num_workers=10,
            use_filesys=config.get("data_use_filesys", True),
            n_clutters=config.get("clut_mnist_clutters"),
            data_dir=config.get("clut_mnist_data_dir", None),
            train_transform=train_transform,
            test_transform=test_transform,
            n_samples_train=config.get("clut_mnist_train_samples"),
            n_samples_test=int(0.15 * config.get("clut_mnist_train_samples")))
        super().__init__(data_fac, device)

    @staticmethod
    def apply_defaults(overrides=None):
        return {**{
            'clut_mnist_train_samples':60021,
            'clut_mnist_clutters': 21,
            'clut_mnist_data_dir': '../../data/cluttered_mnist_60021',
        }, **(overrides if overrides is not None else {})}


class MNISTNormalizedDatasetFactory(MNISTDatasetFactory):
    def __init__(self, config):
        data_resize = config.get("data_resize", (32, 32))
        train_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_transform = Compose([
            Resize(data_resize),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        super().__init__(
            batch_size=config["batch_size"],
            train_transform=train_transform,
            test_transform=test_transform)
