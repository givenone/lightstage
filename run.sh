python binary_reconstruction.py -V -format png -path reconstruction_input/
echo "reconstruction done"
python generate_pointcloud_ns.py reconstruction_output/diffuse_albedo.png dist0.exr reconstruction_output/syn.tif result/before.ply
echo "generating ply file done"
make
./mesh_opt result/before.ply -lambda 0.01 norm:result/fixed.ply
echo "mesh & normal combination done"