# lightstage

- binary spherical gradient pattern reconstruction
- generating ply file
- normal, poisition combination

## How to use
* you need to install suitesparse library to run. 

* edit or create .fc file to run. .fc file contains (fx, fy, width/2(center_x), height/2(center_y)). Take notice of sign. The sign and magnitude (axis) should match with those of the normals we obtained using light stage.
  
* build :

```
    make
```

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

* mesh_opt : follow instruction in [link](https://github.com/givenone/normal_position_combination). You can use .fc files (range grid) or you can just run mesh using trinangulation. As documented on the link above, using it generates much qualified results.

```
./mesh_opt result/before.ply -fc emily.fc -lambda 0.01 norm:result/fixed.ply
./mesh_opt result/before.ply -lambda 0.01 norm:result/fixed.ply
./mesh_opt result/before.ply -lambda 0.01 result/fixed.ply
```

- you may change "-lambda" value change weights for optimization. Also, you may get rid of "norm" flag to save only point cloud data, no normals.