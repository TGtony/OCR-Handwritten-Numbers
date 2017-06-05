/******

    Created by Tony Gong

    p2.cpp

    Tests training data on simple test images.

    ./p2 <test image> <label file .txt> <train data .txt>

    ***IMPORTANT***: suggested to use label and training files generated
                     under a counter of 10 from training since these
                     test images provided does not contain symbols.

******/

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>
#include<string>
#include<sstream>
#include<fstream>

#include "ocr.h"

using namespace std;
using namespace cv;
using namespace ml;

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cout << "Usage: " << argv[0] << " <test image> <label file .txt> <train data .txt>" << endl;
        return 0;
    }

    const string testing_image(argv[1]);
    const string label_file(argv[2]);
    const string train_file(argv[3]);

    OCR ocr;
    Ptr<KNearest> kNearest(KNearest::create());

    FileStorage labels_input(label_file, FileStorage::READ);
    if (labels_input.isOpened() == false)
    {
        cout << "Unable to open file: " << label_file << endl;
        return(0);
    }
    FileStorage train_input(train_file, FileStorage::READ);
    if (train_input.isOpened() == false)
    {
        cout << "Unable to open file: " << train_file << endl;
        return(0);
    }
    ocr.read_inputs(kNearest, labels_input, train_input);

    Mat test_image = imread(testing_image); // open test image
    if (test_image.empty()) {
        cout << "Unable to open image." << testing_image << endl;
        return(0);
    }
    // pre-processing
    Mat test_thresh = ocr.pre_processing(test_image);

    // clone
    Mat test_thresh2 = test_thresh.clone();

    // find contours of each number
    vector<vector<Point>> contour_vector;
    vector<Vec4i> hierarchy;
    findContours(test_thresh2, contour_vector, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // sort contours from left to right, top to bottom
    sort(contour_vector.begin(), contour_vector.end(), contour_sorter());

    for(size_t i = 0; i < contour_vector.size(); i++)
    {
        if(ocr.check_contour(contour_vector, i)==true)
        {
            // get bounding rectangle
            Rect rect_num = boundingRect(contour_vector[i]);
            rectangle(test_image, rect_num, Scalar(255, 0, 0), 3);

            // resize and convert character for use in kNN
            Mat detected_character = ocr.find_character(test_thresh, contour_vector, i);

            // stores result of kNN
            Mat detected_character2(0, 0, CV_32FC1);

            // kNN, returns the value of the number
            kNearest->findNearest(detected_character, 3, detected_character2);
            int ascii_value = detected_character2.at<float>(0,0);
            char c = ocr.convert_to_char(ascii_value);
            cout << c << endl;

            // image of current number
            Mat number = test_image(rect_num);

            // show images
            namedWindow("Number2", CV_WINDOW_NORMAL);
            namedWindow("Number", CV_WINDOW_NORMAL);
            imshow("Number2", number);
            imshow("Number", test_image);
            waitKey(0);
        }
    }
    return 0;
}

