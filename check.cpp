#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int image_height, image_width;

Mat normal_map;
Mat depth_map;
Mat displacement_map;

float cx, cy, fx, fy;

int read_normal_map(string addr)
{
    // How to read (using opencv imread function)

    normal_map = imread(addr, IMREAD_UNCHANGED);
    if(normal_map.empty()) return -1;

    image_height = normal_map.rows;
    image_width = normal_map.cols;

    assert(normal_map.channels() == 3);

    printf("height : %d\nwidth : %d\n", image_height, image_width);

    // for check
    // first channel : nx
    // second channel : ny
    // third channel : nz
    // In most cases, nz should be positive and largest
    
    cout << normal_map.at<Vec3f>(2000, 1000) << endl;
    return 1;
}

int main()
{
    // LSMK ..... : path to normal map (.tif format) 
    // can read into matrix using opencv imread function.
    // See read_normal map function for test. 
    if(read_normal_map("LSMK_0603_103015/Front/result/normal.tif") == -1) cout << "error in normal map" <<endl;
}