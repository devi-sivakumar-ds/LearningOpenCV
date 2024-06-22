// FaceDetection.cpp : Defines the entry point for the application.
#include<stdlib.h>
#include<string.h>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<opencv2\objdetect\objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void detectFaces(Mat img, char output_path[]);
CascadeClassifier faceDetection;
char input_path[100];


int main() {

	if (!faceDetection.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")) {
		cout << "\n XML File not found";
		exit(0);
	}

	Mat image1 = imread("H:/pic1.jpg");

	if (image1.empty()) {
		cout << "\n image is not loaded! - ";
	}

	char output_path1[100];
	strcpy(output_path1, "H:\\pic1final.jpg");
	detectFaces(image1, output_path1);

	return 0;
}

void detectFaces(Mat img, char output_path[]) {
    vector<Rect> faces;

    // Detect faces
    faceDetection.detectMultiScale(img, faces);

    // Create a mask for the faces
    Mat mask = Mat::zeros(img.size(), img.type());
    for (int i = 0; i < faces.size(); i++) {
        rectangle(mask, faces[i], Scalar(255, 255, 255), FILLED);
    }

    // Blur the entire image
    Mat blurred_image;
    GaussianBlur(img, blurred_image, Size(21, 21), 0);

    // Combine the blurred background with the original face
    Mat result;
    bitwise_and(img, mask, result);
    bitwise_and(blurred_image, Scalar(255, 255, 255) - mask, blurred_image);
    add(result, blurred_image, result);

    imwrite(output_path, result);
    String windowName = "My Face Detection Window";
    namedWindow(windowName);
    imshow(windowName, result);
    waitKey(0);
    destroyWindow(windowName);

    cout << "\n Image is detected and processed successfully!";
}