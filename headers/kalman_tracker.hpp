#include <opencv2/opencv.hpp>
#include <detection.hpp>
#include <toml.hpp>

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
    int 
        classId,
        nomatch,
        maxnomatch,
        maxlen,
        id;
    bool 
        updated=false,
        todelete;
    list<Point2d> Points;
    Mat prev_hist;
    void Update(Detection &d, float dt) {
        updated=true;
        auto b = d.get_center();
        Mat meas = (Mat_<float>(2,1) << b.x, b.y);
        // cout << kf.measurementMatrix; 
        // cout<< " here we go"<< endl << endl;
        kf.correct(meas); 
        // cout<<  " again" << endl;
        kf.transitionMatrix.at<float>(0,2) = dt; 
        kf.transitionMatrix.at<float>(1,3) = dt;
        //cout << kf.statePost << endl;
        Points.push_front(statetoPoint2d(kf.statePost));
        if (Points.size()>maxlen)
            Points.pop_back();
        kf.predict();
    };
    void Update(float dt) {
        updated=false;
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
        nomatch =0;
        classId = det.classId;
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
        // cout << "State:\n" << kf.statePre <<endl;
        kf.statePre = kf.statePost;
        // cout << kf.statePre << endl;
        Points.push_back(b);
    }
    bool operator==(const Track& b) {
        return id==b.id,
               nomatch==b.nomatch,
               classId ==b.classId;
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
        circle(img, p, 2, CV_RGB(255,0 , 0), 2);
        line(img,prev, p,CV_RGB(225, 0, 0) , 1);
        prev=p;
    }
}

int Track::id_n;

class KalmanTracker{
    private:
    Mat calcHistRGB(Mat);
    void Register(list<Detection>, Mat&);
    list<Line> DetLines;

    public:
    int 
        maxNoMatch,
        maxPointsCount;
    float
        // @TODO: Treshold distance from state params
        tresholdDist,
        histTreshold;
    list<Track> Tracks;
    
    void DrawCV(Mat&, bool);
    void UpdateConfig(string path_to_config);
    void Update(list<Detection>, Mat& , float);
    KalmanTracker(int nomatch = 25,int maxpoints = 25 ,float dist = 100, float hist_tr =0.6):
        maxNoMatch(nomatch),
        maxPointsCount(maxpoints),
        tresholdDist(dist),
        histTreshold(hist_tr)
        {}

    void RemoveOldTracks() {
        Tracks.remove_if([=](Track tr){
            if (tr.nomatch>=this->maxNoMatch){
                cout << "removing track with id " << tr.id <<endl;
                for (auto p: tr.Points)
                    cout<<p<<"-";
                cout<<"\n";
                return true;
            }
            return false;
        });
    };
};

// @TODO: should be made with grpc updating
void KalmanTracker::UpdateConfig (string path_to_config) {
    toml::value config;
    int distT,pointsC, nomatch;
    float histT;
    try {
        config=toml::parse(path_to_config);
        distT =config["tracker"]["distTreshold"].as_integer();
        histT =config["tracker"]["histTreshold"].as_floating();
        pointsC=config["tracker"]["pointsInTrack"].as_integer();
        nomatch=config["tracker"]["maxNoMatch"].as_integer();
        if (distT<0 or histT<0 or pointsC<0 or nomatch<0) {
            throw invalid_argument("One of tracker params is negative value");
        }
        auto lines = config["tracker"]["lines"].as_array();
        for (auto &l : lines ) {
            DetLines.push_back(l.as_table());    
        }
    } catch(const exception &e) {
        cout << e.what() << "\nUsing default configuration!\n";
        exit(-1);
    }
    this->histTreshold = histT;
    this->maxNoMatch = nomatch;
    this->maxPointsCount = pointsC;
    this->histTreshold = histT;
};

void KalmanTracker::DrawCV(Mat &img, bool with_det_lines=true) {
    for (auto &t: this->Tracks){
        t.DrawCV(img);
    }
    if (with_det_lines) {
        for (auto &l: this->DetLines) {
            l.DrawCV(img);
        }
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
            cout<< "New track from: " << d.get_center() << endl;
            auto NewTr = Track(d, this->maxPointsCount, this->maxNoMatch);
            NewTr.prev_hist = calcHistRGB(img(d.bbox));
            this->Tracks.push_back(move(NewTr));
        }
    }
}



void KalmanTracker::Update(list<Detection> dets, Mat &img, float dt) {
    // cout << "\nUPD\n";
    if (dets.size() == 0) {
        for (auto &tr: this->Tracks) {
            tr.Update(dt);
        }
        this->RemoveOldTracks();
        return;
    }
    
    if (this->Tracks.size()==0) {
        this->Register(move(dets), img);
        return;
    }
    // auto histMap = make_unique<unordered_map<Detection, Mat>>();
    unique_ptr<unordered_map<Detection, Mat>> histMap(new unordered_map<Detection, Mat>);
    for (auto& tr : this->Tracks) {
        auto track_p = tr.Points.front();
        auto best_hist_score = histTreshold;
        auto best_det = dets.end();
        for (auto d = dets.begin(); d!=dets.end(); d++) {
            if (tr.classId != d->classId) continue;
            if (d->appended) {
                continue;
            }
            if (dst(track_p, d->get_center()) <= this->tresholdDist) {
                // cout<<"Appending to track "<<tr.id << " "<<  d->get_center() <<endl;
                if (histMap->count(*d)==0) {
                    (*histMap)[*d] = calcHistRGB(img(d->bbox));
                }
                // cout << histMap[*d]<<endl;
                auto hist_score = compareHist(tr.prev_hist, (*histMap)[*d], HISTCMP_BHATTACHARYYA);
                if (hist_score < best_hist_score) {
                    best_hist_score = hist_score;
                    best_det = d;
                }/*  else {
                    cout << "Tresholded: "<< hist_score <<">"<< best_hist_score << endl;
                } */
            }
        }
        if (best_det != dets.end()){
            tr.Update(*best_det, dt);
            tr.prev_hist = (*histMap)[*best_det];
            best_det->appended = true;
        } else {
            tr.Update(dt);
        }
        if (tr.updated) {
            auto ln = Line(tr.Points.front(),tr.Points.back());
            for (auto &lin: this->DetLines) {
                if (ln.CrossedInDirection(lin)) {
                    lin.DrawCV(img);
                }
            }
        }
    }
    /* for (auto &d: dets) {
         cout <<"app"<< d.appended<<endl;
    } */
    this->RemoveOldTracks();
    this->Register(move(dets), img);
}
