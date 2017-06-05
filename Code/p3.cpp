/******

    Created by Tony Gong

    p3.cpp

    Tests training data on test images.

    ./p3 <test image> <label file .txt> <train data .txt>

    Extension of p2

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

    Mat test_image2 = test_image.clone(); // make a copy of test image
    Mat test_morph = ocr.morph_image(test_image); // morph image

    // find contours of the numbers
    vector<vector<Point>> contours_num;
    vector<Vec4i> hierarchy_num;
    findContours(test_morph, contours_num, hierarchy_num, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    // sort the contours from left to right, top to bottom
    sort(contours_num.begin(), contours_num.end(), contour_sorter());

    for(size_t x = 0; x < contours_num.size(); x++)
    {
        // if the area of the contour is significant (not noise)
        if(ocr.check_contour(contours_num, x)==true)
        {
            // find bounding rectangle for each number & make it a new image
            Rect rect_num = boundingRect(contours_num[x]);
            Mat number = test_image(rect_num);

            // preprocessing
            Mat number_thresh = ocr.pre_processing(number);
            copyMakeBorder(number_thresh, number_thresh, 50, 50, 50, 50, BORDER_CONSTANT, 0);

            // clone the image of the number, fix skew if necessary
            Mat number_thresh2 = number_thresh.clone();
            ocr.fix_text_skew(number_thresh, number_thresh2);
            number_thresh = number_thresh2.clone();

            // find contours - contours of each digit of the number
            vector<vector<Point> > contour_digits;
            vector<Vec4i> hierarchy_digits;
            findContours(number_thresh2, contour_digits, hierarchy_digits, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            // sort contours from left to right
            sort(contour_digits.begin(), contour_digits.end(), contour_sorter2());

            string detected_line; // string to hold the value of the number
            for(size_t i = 0; i < contour_digits.size(); i++)
            {
                // if the area of the contour is significant
                if(ocr.check_contour(contour_digits, i)==true)
                {
                    // reshape the image of each digit/character for use in kNN
                    Mat detected_character = ocr.find_character(number_thresh, contour_digits, i);

                    // stores result of kNN
                    Mat detected_character2(0, 0, CV_32FC1);

                    // use kNN with k = 3
                    kNearest->findNearest(detected_character, 1, detected_character2);

                    // get value of the digit and append it to the string
                    int ascii_value = detected_character2.at<float>(0,0);
                    char c = ocr.convert_to_char(ascii_value);
                    detected_line = detected_line + c;
                }
            }

            // draw retangle around each digit of the number
            for(size_t i = 0; i < contour_digits.size(); i++)
            {
                if(ocr.check_contour(contour_digits, i)==true)
                {
                    Rect contour_rectangle = boundingRect(contour_digits[i]);
                    rectangle(number_thresh, contour_rectangle, Scalar(255, 0, 0), 2);
                }
            }

            cout << detected_line << endl;

            // draw rectangle around each number in the image
            rectangle(test_image2, rect_num, Scalar(0, 255, 0), 5);

            // show image of digits
            namedWindow( "Detected Characters", CV_WINDOW_AUTOSIZE );
            imshow( "Detected Characters", number_thresh );

            // show image of numbers
            namedWindow( "Original Image", CV_WINDOW_NORMAL );
            imshow( "Original Image", test_image2 );

            waitKey(0);
        }
    }
    return 0;
}
