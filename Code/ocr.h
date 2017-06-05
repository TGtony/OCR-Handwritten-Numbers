/******

    Created by Tony Gong

    ocr.h

    OCR class

******/

#ifndef OCR_H
#define OCR_H

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<vector>
#include<string>
#include<sstream>

using namespace std;
using namespace cv;
using namespace ml;

class OCR
{
    public:

        // pre_processing
        // convert image to grayscale and threshold
        Mat pre_processing(Mat & image)
        {
            Mat to_gray;
            cvtColor(image, to_gray, CV_BGR2GRAY);

            Mat thresh;
            threshold(to_gray, thresh, 100, 255, CV_THRESH_BINARY_INV);

            return thresh.clone();
        }

        // morph_image
        // morphs image to maintain numbers that consists of more than 1 digit
        Mat morph_image(Mat & image)
        {
            Mat to_gray;
            cvtColor(image, to_gray, CV_BGR2GRAY);

            Mat gradient_morph;
            Mat morph_kernel_1 = getStructuringElement(MORPH_RECT, Size(1, 20));
            morphologyEx(to_gray, gradient_morph, MORPH_GRADIENT, morph_kernel_1);

            Mat thresh;
            threshold(to_gray, thresh, 100, 255, CV_THRESH_BINARY_INV);

            Mat morph_2;
            Mat morph_kernel_2 = getStructuringElement(MORPH_RECT, Size(100, 1));
            morphologyEx(thresh, morph_2, MORPH_CLOSE, morph_kernel_2);

            return morph_2.clone();
        }

        // creates a new image of the number using its contours
        // resize to 20x20
        // converts image to format to store and be used with kNN
        Mat find_character(Mat & image, vector<vector<Point>> & contour_vector, int i)
        {
            Rect contour_rectangle = boundingRect(contour_vector[i]);
            Mat character = image(contour_rectangle);
            resize(character, character, Size(20, 20));
            character.convertTo(character, CV_32FC1);
            character = character.reshape(1, 1);

            return character.clone();
        }

        // convert_to_ascii
        // returns ascii value for a character
        int convert_to_ascii(char c)
        {
            int ascii_value = c;

            return ascii_value;
        }

        // convert_to_char
        // returns character of an ascii value
        int convert_to_char(int ascii_value)
        {
            char c = ascii_value;
            return c;
        }

        // add_data
        // add to training data
        void add_data(Mat & character, int ascii_value)
        {
            training_data.push_back(character);
            labels.push_back(ascii_value);
        }

        // check_contour
        // checks if contour is large enough (ignores noise, random dots, stray marks)
        bool check_contour(vector<vector<Point>> & contour_vector, int i)
        {
            if(contourArea(contour_vector[i])>150)
                return true;
            return false;
        }

        // create_outputs
        // outputs training data to txt files
        void create_outputs(FileStorage & labels_output, FileStorage & train_output)
        {
            //FileStorage labels_output(labels, FileStorage::WRITE);
            labels_output << "labels" << labels;
            labels_output.release();

            //FileStorage train_output(train, FileStorage::WRITE);
            train_output << "training" << training_data;
            train_output.release();
        }

        // read_inputs
        // reads txt files with training data and create kNN model
        void read_inputs(Ptr<KNearest> kNearest, FileStorage & labels_input, FileStorage & train_input)
        {
            labels_input["labels"] >> labels;
            labels_input.release();

            train_input["training"] >> training_data;
            train_input.release();

            kNearest->train(training_data, ROW_SAMPLE, labels);
        }

        // fix_text_skew
        // creates a rotated rectangle around the text
        // rotates the rectangle from it's current angle to get a vertical rectangle
        void fix_text_skew( Mat & image, Mat & rotated )
        {
            vector<Point> points;
            Mat_<uchar>::iterator it = image.begin<uchar>();
            Mat_<uchar>::iterator end = image.end<uchar>();
            for (; it != end; ++it)
                if (*it)
                    points.push_back(it.pos());

            RotatedRect box = minAreaRect(Mat(points));

            if(box.angle < -45)
                box.angle += 90;

            Mat rot_mat = getRotationMatrix2D(box.center, box.angle, 1);

            // set size of rectangle so result doesnt get cut off
            int rows = image.rows;
            int cols = image.cols;
            if(rows < cols)
                rows = cols;
            else
                cols = rows;
            warpAffine(image, rotated, rot_mat, Size(rows, cols), INTER_CUBIC);
        }

    private:
        Mat labels;
        Mat training_data;
};

// sorts contours from left to right, top to bottom
struct contour_sorter
{
    bool operator ()( const vector<Point>& a, const vector<Point> & b )
    {
        Rect rectangle_a(boundingRect(a));
        Rect rectangle_b(boundingRect(b));
        return ( (rectangle_a.x + 50*rectangle_a.y) < (rectangle_b.x + 50*rectangle_b.y) );
    }
};

// sorts contours from left to right
struct contour_sorter2
{
    bool operator ()( const vector<Point>& a, const vector<Point> & b )
    {
        Rect rectangle_a(boundingRect(a));
        Rect rectangle_b(boundingRect(b));
        return ( rectangle_a.x < rectangle_b.x );
    }
};

#endif
