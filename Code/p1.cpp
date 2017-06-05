/******

    Created by Tony Gong

    p1.cpp

    Trains dataset and output 2 text files.

    ./p1 <label file .txt> <train data .txt> <counter>

    ***IMPORTANT***: counter is suggested to be either 10 or 15.
                     10 (numbers 0-9)
                     15 (numbers 0-9 and '(', ')', '+', '-', 'x')

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
        cout << "Usage: " << argv[0] << " <label file .txt> <train data .txt> <counter>" << endl;
        return 0;
    }

    const string labels(argv[1]);
    const string training(argv[2]);
    // set counter to 10 if using only 0-9, set to 15 to include the otehr symbols;
    // 10 digits (0-9), 5 symbols( '(', ')', '+', '-', 'x')
    const int counter = atoi(argv[3]);

    OCR ocr;

    // for each image
    for(int w = 0; w < counter; w++)
    {
        Mat train;
        stringstream ss;
        ss << w;
        string str = ss.str();
        string str2 = str+".jpg";
        train = imread(str2);

        if(train.empty())
        {
            cout << "Unable to open image: " << str2 << endl;
            return 0;
        }

        // pre-processing
        Mat processed;
        processed = ocr.pre_processing(train);

        // clone
        Mat processed2;
        processed2 = processed.clone();

        // find contours - each image of the number
        vector<vector<Point>> contour_vector;
        vector<Vec4i> hierarchy;
        findContours(processed2, contour_vector, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        // checks if area of contour is large enough (removes contours from noise)
        // process the image of the number and convert to ascii
        // store the image
        for(size_t i = 0; i < contour_vector.size(); i++)
        {
            if(ocr.check_contour(contour_vector, i)==true)
            {
                Mat found_character = ocr.find_character(processed, contour_vector, i);
                int ascii_value;
                if(w > 9)
                {
                    if(w == 10)
                        ascii_value = ocr.convert_to_ascii('(');
                    if(w == 11)
                        ascii_value = ocr.convert_to_ascii(')');
                    if(w == 12)
                        ascii_value = ocr.convert_to_ascii('+');
                    if(w == 13)
                        ascii_value = ocr.convert_to_ascii('-');
                    if(w == 14)
                        ascii_value = ocr.convert_to_ascii('x');
                }
                else
                {
                    ascii_value = ocr.convert_to_ascii(str[0]);
                }
                ocr.add_data(found_character, ascii_value);
            }
        }

        FileStorage labels_output(labels, FileStorage::WRITE);
        FileStorage train_output(training, FileStorage::WRITE);
        ocr.create_outputs(labels_output, train_output);
    }
    return 0;
}
