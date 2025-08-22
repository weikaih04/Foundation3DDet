# KITTI Object Depth

We provide the scripts to find the KITTI depth annotation for KITTI Object provided by [Omni3D](https://github.com/facebookresearch/omni3d/blob/main/DATA.md).

1. Download the official dataset from [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), including the raw data (about 200G) and fine-grained ground-truth depth maps. 

2. Put everything under `data/kitti_depth` folder. And make sure you already download the and prepare the Omni3D datasets.

3. Generate depth by running `python scripts/kitti/gen_depth.py`.

4. (Optional) If you want to use HDF5 as databackend, you can run:
```bash
cd data
python -m vis4d.data.io.to_hdf5 -p KITTI_object_depth
```

```bash
data
├── omni3d
├── KITTI_object
├── kitti_depth
├── KITTI_object_depth
```