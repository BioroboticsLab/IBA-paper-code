import numpy as np

# from attribution_bottleneck.attribution.base import *
from attribution_bottleneck.utils.transforms import HeatmapTransform
import torch


class GridView:
    """ access something by 2D-tile indices """
    def __init__(self, orig_dim: tuple, tile_dim: tuple):
        self.orig_r = orig_dim[0]
        self.orig_c = orig_dim[1]
        self.tile_h = tile_dim[0]
        self.tile_w = tile_dim[1]
        self.tiles_r = self.orig_r // self.tile_h
        self.tiles_c = self.orig_c // self.tile_w
        self.grid = (self.tiles_r, self.tiles_c)

        if self.orig_r % self.tile_h != 0 or self.orig_c % self.tile_w != 0:
            print("Warning: GridView is not sound")

    def tile_slice(self, tile_r: int, tile_c: int):
        """ get the slice that would return the tile r,c """
        assert tile_r < self.tiles_r, \
            "tile {} is out of range with max {}".format(tile_r, self.tiles_r)
        assert tile_c < self.tiles_c, \
            "tile {} is out of range with max {}".format(tile_c, self.tiles_c)

        r = tile_r * self.tile_h
        c = tile_c * self.tile_w

        if tile_r == self.tiles_r - 1:
            slice_r = slice(r, None)
        else:
            slice_r = slice(r, r + self.tile_h)

        if tile_c == self.tiles_c - 1:
            slice_c = slice(c, None)
        else:
            slice_c = slice(c, c + self.tile_w)

        return slice_r, slice_c


class NumpyGridView(GridView):
    """ access an image by 2D-tile indices """
    def __init__(self, img: np.ndarray, tile_dim: tuple):
        assert len(img.shape) == 3, "pass numpy images w/o batch dim and with color channel!"
        self.img = img
        super().__init__(img.shape[:-1], tile_dim)

    def get_tile(self, tile_r: int, tile_c: int):
        """ get content of tile r, c """
        slices = self.tile_slice(tile_r, tile_c)
        return self.img[slices]

    def set_tile(self, tile_r: int, tile_c: int, origin: np.ndarray):
        slices = self.tile_slice(tile_r, tile_c)
        self.img[slices] = origin[slices]

    def get_tile_means(self):
        means = np.zeros((self.tiles_r, self.tiles_c))
        for r in range(self.tiles_r):
            for c in range(self.tiles_c):
                means[r, c] = np.mean(self.img[self.tile_slice(r, c)])
        return means

    def shuffled(self):
        ir = np.linspace(start=0, stop=self.tiles_r, num=self.tiles_r, endpoint=False).astype(int)
        ic = np.linspace(start=0, stop=self.tiles_c, num=self.tiles_c, endpoint=False).astype(int)
        idx_r, idx_c = np.meshgrid(ir, ic)
        np.random.shuffle(idx_r.reshape(-1))
        np.random.shuffle(idx_c.reshape(-1))
        shuffled = self.img.copy()
        for r in range(self.tiles_r):
            for c in range(self.tiles_c):
                shuffled[self.tile_slice(r, c)] = self.img[
                    self.tile_slice(idx_r[r, c], idx_c[r, c])]
        return shuffled


class Perturber:
    def perturbe(self, r: int, c: int):
        """ perturbe a tile or pixel """
        raise NotImplementedError

    def get_current(self) -> np.ndarray:
        """ get current image with some perturbations """
        raise NotImplementedError

    def get_idxes(self, hmap: np.ndarray, reverse=False) -> list:
        """return a sorted list with shape length NUM_CELLS of
        which pixel/cell index to blur first"""
        raise NotImplementedError

    def get_grid_shape(self) -> tuple:
        """ return the shapeof the grid, i.e. the max r, c values """
        raise NotImplementedError


class PixelPerturber(Perturber):
    def __init__(self, inp: torch.Tensor, baseline: torch.Tensor):
        self.current = inp.clone()
        self.baseline = baseline

    def perturbe(self, r: int, c: int):
        self.current[:, :, r, c] = self.baseline[:, :, r, c]

    def get_current(self) -> torch.Tensor:
        return self.current

    def get_grid_shape(self) -> tuple:
        return self.current.shape

    def get_idxes(self, hmap: np.ndarray, reverse=False) -> list:
        return HeatmapTransform.to_index_list(hmap, reverse)


class GridPerturber(Perturber):
    def __init__(self, original: torch.Tensor, baseline: torch.Tensor, tile_dim):
        assert original.device == baseline.device
        assert len(tile_dim) == 2
        self.view = GridView(tuple(original.shape[-2:]), tile_dim)
        self.current = original.clone()
        self.baseline = baseline
        # print("original shape ", self.current.img.shape)
        # print("tile grid {}x{}".format(self.current.tiles_c, self.current.tiles_r))
        # print("tile size {}x{}".format(self.current.tile_h, self.current.tile_w))

    def perturbe(self, r: int, c: int):
        slc = self.view.tile_slice(r, c)
        self.current[:, :, slc[0], slc[1]] = self.baseline[:, :, slc[0], slc[1]]

    def get_current(self) -> torch.Tensor:
        return self.current

    def get_grid_shape(self) -> tuple:
        return self.view.tiles_r, self.view.tiles_c

    def get_tile_shape(self) -> tuple:
        return self.view.tile_h, self.view.tile_w

    def get_idxes(self, hmap: np.ndarray, reverse=False) -> list:
        grid_hmap = NumpyGridView(np.expand_dims(hmap, 2), self.get_tile_shape()).get_tile_means()
        return HeatmapTransform.to_index_list(grid_hmap, reverse=reverse)
