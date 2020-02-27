#include <opencv2/opencv.hpp>
#include <detection.h>

using namespace std;
using namespace cv;


class Track {
    private:
    static int id_n;
    KalmanFilter kf;
    void initKalman() {
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
        setIdentity(kf.processNoiseCov, cv::Scalar(1e-3));
        // Measures Noise Covariance Matrix R
        setIdentity(kf.measurementNoiseCov, cv::Scalar(0.75)); 
    }
    Point2d statetoPoint2d(Mat st) {
        return Point2d(st.data[0], st.data[1]);
    };

    public:
    int classId;
    int maxlen;
    int nomatch, maxnomatch;
    int id;
    unique_ptr<list<Point2d>> Points;
    Mat prev_hist;

    void Update(Point2d b, int64 dt) {
        kf.correct((Mat_<float>(2,1) << b.x, b.y)); 
        kf.transitionMatrix.at<float>(0,2) = dt; 
        kf.transitionMatrix.at<float>(1,3) = dt;
        kf.predict();
        Points->push_front(statetoPoint2d(kf.statePost));
        if (Points->size()>maxlen)
            Points->pop_back();
    };

    void Update(int64 dt) {
        nomatch++;
        if (nomatch>0);
        kf.transitionMatrix.at<float>(0,2) = dt; 
        kf.transitionMatrix.at<float>(1,3) = dt;
        kf.predict();
        Points->push_front(statetoPoint2d(kf.statePost));
        if (Points->size()>maxlen)
            Points->pop_back();
    };
    // Track(Detection det, int maxl = 15, int maxm = 10): maxlen(maxl), maxnomatch(maxm) {
        // auto b = det.get_center();
        // id = ++id_n;
        // initKalman();
        // kf.statePost = Mat_<float>(4,1) << b.x, b.y, 0., 0.;
        // Points->push_front(b);
    // }
    ~Track() {
    }
};

class KalmanTracker{
    private:
    void Register(list<Detection>);
    void AllDissapear();
    
    public:
    int maxNoMatch;
    int maxBlobs;
    float tresholdDist;
    list<Track> Tracks;
    
    void Update(list<Detection>, Mat& , int64);
    KalmanTracker(int nomatch = 15,int maxblobs = 10 ,float dist = 15.): maxNoMatch(nomatch), maxBlobs(maxblobs), tresholdDist(dist){}
};

auto dst = [](Point2d p1,Point2d p2) {return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));};

void KalmanTracker::Register(list<Detection> dets) {

}
void KalmanTracker::AllDissapear() {

}

void KalmanTracker::Update(list<Detection> dets, Mat &img, int64 dt) {
    if (dets.size() == 0) {
        this->AllDissapear();
        return;
    }
    if (this->Tracks.size()==0) {
        this->Register(move(dets));
        return;
    }
    unordered_map<Detection, Mat> histMap;
    // for(auto& det: dets) {
        // auto minDist = this->tresholdDist;
        // auto bestCand = this->Tracks.end();
        // for(auto trit = this->Tracks.begin();trit!=this->Tracks.end(); trit++){
            // if (det.classId != trit->classId) 
                // continue;
            // if (dist<minDist) {
                // bestCand = trit;
                // minDist = dist;
            // }
        // }
        // if (bestCand != this->Tracks.end()) {
            // bestCand->Update(det, );
            // bestCand->nomatch = 0;
            // det.exist = true;
        // } 
    // }

    for (auto &tr : this->Tracks) {
        auto track_p = tr.Points->front();
        for (auto& d: dets) {
            auto dist = dst(track_p, d.get_center());
        }

    }

}  

Mat calcHistRGB(Mat img) {
    MatND hist;
    int imgCount = 1;
    int dims = 2;
    const int sizes[] = {256,256,256};
    const int channels[] = {0,1,2};
    float rRange[] = {0,256};
    float gRange[] = {0,256};
    float bRange[] = {0,256};
    const float *ranges[] = {rRange,gRange,bRange};
    Mat mask = Mat();
    calcHist(&img, imgCount, channels, mask, hist, dims, sizes, ranges);
    return hist;
}

void testHist(Mat img){
    // auto res = compareHist(hist1, hist2, HISTCMP_BHATTACHARYYA); 
    // cout << res << endl;
    
    // auto img2 = imread("/home/lbstr/Downloads/Telegram Desktop/1.jpg");
    auto img1 = imread("./car1_1.png");
    auto img2 = imread("./car1.png");


    // imshow("img1", img1);
    // imshow("imgq1", eqim1);
    // waitKey(0);

    auto hist1 = calcHistRGB(img1);
    auto hist2 = calcHistRGB(img2);

    cout << compareHist(hist2, hist1, HISTCMP_INTERSECT) << endl;
    cout << compareHist(hist1, hist1, HISTCMP_INTERSECT) << endl;
    // imshow("h2", img(Rect(300, 300, 300, 300)));
    // auto res = compareHist(hist1, hist2, HISTCMP_BHATTACHARYYA); 
    // cout << res << endl;
}