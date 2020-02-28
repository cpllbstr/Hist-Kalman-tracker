#include <opencv2/opencv.hpp>
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
        return Point2d(st.data[0], st.data[1]);
    };

    public:
    KalmanFilter kf;
    void DrawCV(Mat&);
    int 
        classId,
        nomatch,
        maxnomatch,
        maxlen,
        id;
    bool 
        updated,
        todelete;
    list<Point2d> Points;
    Mat prev_hist;
    void Update(Detection &d, int64 dt) {

        auto b = d.get_center();
        Mat meas = (Mat_<float>(2,1) << b.x, b.y);
        cout << kf.measurementMatrix; 
        // cout<< " here we go"<< endl << endl;
        kf.correct(meas); 
        // cout<<  " again" << endl;
        kf.transitionMatrix.at<float>(0,2) = dt; 
        kf.transitionMatrix.at<float>(1,3) = dt;
        kf.predict();
        Points.push_front(statetoPoint2d(kf.statePost));
        if (Points.size()>maxlen)
            Points.pop_back();
    };
    void Update(int64 dt) {
        nomatch++;
        if (nomatch>maxnomatch){
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
        classId = det.classId;
        auto b = det.get_center();
        kf = KalmanFilter(4,2);
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
        // initKalman();
        kf.statePost = Mat_<float>(4,1) << b.x, b.y, 0., 0.;
        Points.push_back(b);
    }
    bool operator==(const Track& b) {
        return id==b.id,
               nomatch==b.nomatch,
               classId ==b.classId;
    }
    ~Track() {
    }
};

void Track::DrawCV(Mat &img) {
    cout<<"track with lenght:"<<this->Points.size()<<"C"<<endl;
    Point prev = Points.front();
    for(auto p: Points) {
        circle(img, p, 6, Scalar(0,230,230), 2);
        line(img,prev, p, CV_RGB(225, 0, 0), 2);
        prev=p;
    }
}

int Track::id_n;

class KalmanTracker{
    private:
    Mat calcHistRGB(Mat);
    void Register(list<Detection>, Mat&);
    
    public:
    int 
        maxNoMatch,
        maxPointsCount;
    float
        // @TODO: Treshold distance from state params
        tresholdDist,
        histTreshold;
    list<Track> Tracks;
    
    void DrawCV(Mat&);
    void Update(list<Detection>, Mat& , int64);
    KalmanTracker(int nomatch = 15,int maxblobs = 10 ,float dist = 15., float histtr =0.6): 
        maxPointsCount(maxblobs),
        tresholdDist(dist),
        histTreshold(histtr)
        {}
};

void KalmanTracker::DrawCV(Mat &img) {
    for (auto t: this->Tracks){
        t.DrawCV(img);
    }
}


Mat KalmanTracker::calcHistRGB(Mat img) {
    MatND hist;
    const int imgCount = 1;
    const int dims = 2;
    const int sizes[] = {256,256,256};
    const int channels[] = {0,1,2};
    const float rRange[] = {0,256};
    const float gRange[] = {0,256};
    const float bRange[] = {0,256};
    const float *ranges[] = {rRange,gRange,bRange};
    const Mat mask = Mat();
    calcHist(&img, imgCount, channels, mask, hist, dims, sizes, ranges);
    return hist;
}

constexpr auto dst = [](Point2d p1,Point2d p2) {return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));};

void KalmanTracker::Register(list<Detection> dets, Mat &img) {

    for (auto &d: dets) {   
        if (!d.appended) {
            auto NewTr = Track(d, this->maxPointsCount, this->maxNoMatch);
            NewTr.prev_hist = calcHistRGB(img(d.bbox));
            this->Tracks.push_back(move(NewTr));
        } else {
            d.appended=false;
        }
    }
}


void KalmanTracker::Update(list<Detection> dets, Mat &img, int64 dt) {
    if (dets.size() == 0) {
        for (auto &tr: this->Tracks) {
            tr.Update(dt);
            if(tr.nomatch>maxNoMatch) {
                this->Tracks.remove(tr);
            }
        }
        return;
    }
    
    if (this->Tracks.size()==0) {
        this->Register(move(dets), img);
        return;
    }
    unordered_map<Detection, Mat> histMap;
    for (auto &tr : this->Tracks) {
        auto track_p = tr.Points.front();
        auto best_hist_score = histTreshold;
        Detection *best_det;
        for (auto& d: dets) {
            if (tr.classId != d.classId) continue;
            if (dst(track_p, d.get_center()) <= this->tresholdDist) {
                if (histMap.find(d) == histMap.end()) {
                    histMap[d] = calcHistRGB(img(d.bbox));
                }
                auto hist_score = compareHist(tr.prev_hist, histMap[d], HISTCMP_BHATTACHARYYA);
                if (hist_score < best_hist_score) {
                    best_hist_score = hist_score;
                    best_det = &d;
                }
            }
        }
        if (best_det != nullptr){
            // cout<< " here we go" << endl;

            //Update with appending new point
            tr.Update(*best_det, dt);
            // cout<<  " again" << endl;
            // cout<<tr.prev_hist.size << endl;
            tr.prev_hist = histMap[*best_det];
            best_det->appended = true;
        } else {
            //Update with no match
            tr.Update(dt);
            if(tr.nomatch>maxNoMatch) {
                this->Tracks.remove(tr);
            }
        }
    }
    this->Register(move(dets), img);
}


/* void testHist(Mat img){
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
} */