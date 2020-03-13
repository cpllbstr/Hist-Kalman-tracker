#pragma once
#include <opencv2/opencv.hpp>
#include <toml.hpp>
#include <detection.hpp>

using namespace std;
using namespace cv;

class Track {
private:
    static int id_n;
    void initKalman() {
        // Transition State Matrix A
        // Note: set dT at each processing step!
        // [ 1 0 dT  0 ]
        // [ 0 1  0 dT ]
        // [ 0 0  1  0 ]
        // [ 0 0  0  1 ]
        setIdentity(this->kf.transitionMatrix);
        // Measure Matrix H
        // [ 1 0 ]
        // [ 0 1 ]
        setIdentity(this->kf.measurementMatrix);
        // Process Noise Covariance Matrix Q
        // [ Ex   0   0     0    ]
        // [ 0    Ey  0     0    ]
        // [ 0    0   Ev_x  0    ]
        // [ 0    0   0     Ev_y ]
        setIdentity(this->kf.processNoiseCov, cv::Scalar(1e-5));
        // Measures Noise Covariance Matrix R
        setIdentity(this->kf.measurementNoiseCov, cv::Scalar(1e-2));
        // setIdentity(this->kf.errorCovPost, Scalar::all(1e-2));
    }
    Point2d statetoPoint2d(Mat st) {
        // cout<< st << endl;
        // cout << st.at<float>(0) <<"-"<< st.at<float>(1) << endl;
        return Point2i(int(st.at<float>(0)), int(st.at<float>(1)));
    };
public:
    KalmanFilter kf;
    void DrawCV(Mat&);
    int nomatch, maxnomatch, maxlen, id;
    bool updated=false, todelete;
    list<Point2d> Points;
    Detection prev_det;
    Mat prev_hist;
    void Update(Detection &d, float dt) {
        updated=true;
        this->prev_det = d;
        auto b = d.get_center();
        Mat meas = (Mat_<float>(2,1) << b.x, b.y);
        kf.correct(meas);
        kf.transitionMatrix.at<float>(0,2) = dt;
        kf.transitionMatrix.at<float>(1,3) = dt;
        Points.push_front(statetoPoint2d(kf.statePost));
        if (Points.size()>maxlen)
            Points.pop_back();
        kf.predict();
    };
    void Update(float dt) {
        updated=false;
        nomatch++;
        if (nomatch>maxnomatch) {
            todelete = true;
            return;
        }
        kf.transitionMatrix.at<float>(0,2) = dt;
        kf.transitionMatrix.at<float>(1,3) = dt;
        kf.predict();
        Points.push_front(statetoPoint2d(kf.statePost));
        if (Points.size()>maxlen)
            Points.pop_back();
    };
    Track(Detection &det, int maxle = 15, int maxno = 10): maxlen(maxle), maxnomatch(maxno) {
        id = ++id_n;
        nomatch =0;
        prev_det = det;
        auto b = det.get_center();
        kf = KalmanFilter(4,2);
        // Transition State Matrix A
        // Note: set dT at each processing step!
        // [ 1 0 dT  0 ]
        // [ 0 1  0 dT ]
        // [ 0 0  1  0 ]
        // [ 0 0  0  1 ]
        setIdentity(kf.transitionMatrix);
        // Measure Matrix H
        // [ 1 0 ]
        // [ 0 1 ]
        setIdentity(kf.measurementMatrix);
        // Process Noise Covariance Matrix Q
        // [ Ex   0   0     0    ]
        // [ 0    Ey  0     0    ]
        // [ 0    0   Ev_x  0    ]
        // [ 0    0   0     Ev_y ]
        setIdentity(kf.processNoiseCov, cv::Scalar(1e-5));
        // Measures Noise Covariance Matrix R
        setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-2));
        kf.statePost = (Mat_<float>(4,1) << b.x, b.y, 0., 0.);
        kf.statePre = kf.statePost;
        Points.push_back(b);
    }
    bool operator==(const Track& b) {
        return id==b.id,
               nomatch==b.nomatch,
               prev_det.bbox == b.prev_det.bbox;
    }
    ~Track() {
        // Points.~list();
    }
};

void Track::DrawCV(Mat &img) {
    // cout<<"track "<< this->id<<" with lenght:"<< this->Points.size() <<" points"<< this->Points.front()<< this->Points.back()<< endl;
    /* for(auto p: Points){
        cout << p <<"-";
    } */
    // cout <<endl;
    Point prev = Points.front();
    putText(img, "ID:"+to_string(id),prev,FONT_HERSHEY_SIMPLEX, 0.25, CV_RGB(250,230,0),1.5);
    for(auto p: Points) {
        circle(img, p, 2, CV_RGB(255,0, 0), 2);
        line(img,prev, p,CV_RGB(225, 0, 0), 1);
        prev=p;
    }
}

int Track::id_n;
