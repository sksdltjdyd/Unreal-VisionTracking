#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

///////// Images ////////

void main() {
	string path = "C:/Users/User/Documents/Git/Unreal-VisionTracking/Resources/Image1.jpg";
	Mat img = imread(path);
	imshow("Image", img);
	waitKey(0);
}