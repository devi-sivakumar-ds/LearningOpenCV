#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void detectAndBeautifyFace(Mat img, const char* output_path);
CascadeClassifier faceDetection;

int main() {
    if (!faceDetection.load("C:\\opencv\\sources\\data\\haarcascades\\haarcascade_frontalface_default.xml")) {
        cout << "\n XML File not found";
        return -1;
    }

    Mat image1 = imread("H:/pic1.jpg");
    if (image1.empty()) {
        cout << "\n Image not loaded!";
        return -1;
    }

    const char* output_path1 = "H:\\pic1_beautified.jpg";
    detectAndBeautifyFace(image1, output_path1);

    return 0;
}

void detectAndBeautifyFace(Mat img, const char* output_path) {
    vector<Rect> faces;
    Mat gray;

    // Converting to grayscale for face detection
    cvtColor(img, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    faceDetection.detectMultiScale(gray, faces);

    for (size_t i = 0; i < faces.size(); i++) {
        Mat faceROI = img(faces[i]);

        // Skin Smoothing using Bilateral Filter
        Mat smoothed;
        bilateralFilter(faceROI, smoothed, 15, 80, 80);

        // Brightness and Contrast Adjustment
        Mat brightened;
        smoothed.convertTo(brightened, -1, 1.2, 10);

        // Blemish Removal (Optional: using inpainting)
        Mat mask = Mat::zeros(faceROI.size(), CV_8U);

        // Replace the original face region with the filtered face
        brightened.copyTo(img(faces[i]));
    }

    imwrite(output_path, img);
    String windowName = "Face Beautification";
    namedWindow(windowName);
    imshow(windowName, img);
    waitKey(0);
    destroyWindow(windowName);

    cout << "\n Image processed and saved successfully!";
}