#pragma once
#include <opencv2/opencv.hpp>
#include <detection.hpp>
#include <toml.hpp>
#include <yolo_grpc_impl.h>
#include <track.hpp>
#include <thread>

struct EnvVarException : public exception {
    const char* what() const throw() {
        return "DETECTOR_ADDR_PORT enviroment variable is not set! Run command in shell: export DETECTOR_ADDR_PORT=\"127.0.0.1:8000\"";
    }
};

class KalmanTracker {
private:
    unique_ptr<STYoloClient> s;
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
    void Update(list<Detection>, Mat&, float);
    KalmanTracker(int nomatch = 25,int maxpoints = 25,float dist = 100, float hist_tr =0.6):
        maxNoMatch(nomatch),
        maxPointsCount(maxpoints),
        tresholdDist(dist),
        histTreshold(hist_tr)
    {
        const auto ip_addr = getenv("DETECTOR_ADDR_PORT");
        if (ip_addr == NULL) {
            throw EnvVarException();
        }
        this->s = unique_ptr<STYoloClient>(new STYoloClient(grpc::CreateChannel(ip_addr, grpc::InsecureChannelCredentials())));
    }

    void RemoveOldTracks() {
        Tracks.remove_if([=](Track tr) {
            if (tr.nomatch>=this->maxNoMatch) {
                // cout << "removing track with id " << tr.id <<endl;
                // for (auto p: tr.Points)
                // cout<<p<<"-";
                // cout<<"\n";
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
    for (auto &t: this->Tracks) {
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

constexpr auto dst = [](Point2d p1,Point2d p2) {
    return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
};

void KalmanTracker::Register(list<Detection> dets, Mat &img) {
    for (auto &d: dets) {
        if (!d.appended) {
            // cout<< "New track from: " << d.get_center() << endl;
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
            if (tr.prev_det.classId != d->classId) continue;
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
                }
            }
        }
        if (best_det != dets.end()) {
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
                    // cout << "DetL: ";
                    // lin.Print();
                    // cout << "Trak: ";
                    // ln.Print();
                    // ln.DrawCV(img);
                    thread t = thread([&](){
                        s->EndDetection("0",tr, img);
                    }
                    );
                    t.detach();
                    cout << "Crossed!" << endl;
                    // @TODO: here should be sending through gRPC
                }
            }
        }
    }
    this->RemoveOldTracks();
    this->Register(move(dets), img);
}
