# lightstage

- binary spherical gradient pattern reconstruction
- generating ply file
- normal - poisition combination

## How to use

* simply run :
```
   bash run.sh 
```
* You may change some configurations (recon.conf) such as focal length, sensor size.
* binary_reconstruction.py :
```
    python binary_reconstruction.py [-V] [-format "format of input files"]  [-path "path of input files"]
```
* generate_pointcloud :
```
    python generate_pointcloud_ns.py [color file path] [depth map path] [normal map path] [output file path]
```
* mesh_opt : follow instruction in [link](https://github.com/givenone/normal_position_combination)
```
./mesh_opt result/before.ply -lambda 0.01 norm:result/fixed.ply
```
you may change "-lambda" value change weights for optimization. Also, you may get rid of "norm" flag to save only point cloud data, no normals.