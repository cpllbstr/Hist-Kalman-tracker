#include "opencv2/opencv.hpp"
#include <iostream>
#include <unistd.h>
#include "darknet.h"
#include <chrono>
#include <string.h>


using namespace cv;
using namespace std;

//Blob - represnts center of image
class Blob{
    public:
    string name;
    int x, y;
    bool exist;
    string ToJSON();
    Blob() {x=y=0; exist = false;}
    Blob(detection, char*, int, int);
    Blob(int _x, int _y): x(_x),y(_y){
        exist = false;
    }
    Blob(const Blob &p): x(p.x), y(p.y), exist(p.exist) {
        this->name = p.name;
    }
    ~Blob(){}
    Blob& operator=(const Blob& b) {
        this->x = b.x;
        this->y = b.y;
        this->exist = b.exist;
        return *this;
    }
};

Blob::Blob(detection d, char* name, int height, int width) {
    this->x =d.bbox.x * width;
    this->y = d.bbox.y * height;
    this->exist = false;
    this->name = string(name);
}

//Distanse - calculates distanse between to blobs
float Distance(Blob p1, Blob p2) {
    return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}

std::list<Blob> BlobsFromDets(detection *dets, int num, float thresh, char **names, int classes, int height, int width) {
    std::list<Blob> blst;
    for (int i = 0; i<num; ++i) {
        for (int j = 0; j<classes; ++j) {
            if (dets[i].prob[j] > thresh) {
                blst.push_back(Blob(dets[i], names[j], height, width));
            }
        }
    }
    return blst;
}

//Track - stores blobs and predicts next
class Track {
    static int id_n;
    public:
    bool continued;
    int maxblobs;
    int nomatch;
    int id;
    vector<Blob> Blobs;
    void AddBlob(Blob b);
    string ToJSON();
    void DrawCV(Mat &i);
    Track(Blob b) {
        continued = false;
        maxblobs=10;
        id_n++;
        if (id_n>100000) {
            id_n = 0;
        }
        this->id = id_n;
        maxblobs=5;
        Blobs.push_back(b);
    }
    Track(Blob b, int nmax){
        continued = false;
        maxblobs = nmax;
        id_n++;
        if (id_n>100000) {
            id_n = 0;
        }
        this->id = id_n;
        Blobs.push_back(b);
    }
    ~Track() {
        Blobs = vector<Blob>();
        Blobs.~vector();
    }
};

int Track::id_n;

//AddBlob - adds Blob to the begin of Blobs vector
void Track::AddBlob(Blob b){
    int sz = this->Blobs.size();
    if (sz == this->maxblobs) {
        this->Blobs.pop_back();
        this->Blobs.insert(this->Blobs.begin(), b);
    } else {
        this->Blobs.insert(this->Blobs.begin(), b);
    }
    this->continued = true;
    return;
}

void Track::DrawCV(Mat &img) {
    auto color = Scalar(0,0,255);
    cout<<"track with lenght:"<<this->Blobs.size()<<"C"<<this->continued<<endl;
    for(auto b = this->Blobs.begin(); b!=this->Blobs.end(); ++b) {
        circle(img, Point2d(b->x, b->y), 6, Scalar(0,230,230), 2);
        if (b!= this->Blobs.begin()){
            auto prev=b-1;
            line(img, Point2d(prev->x, prev->y), Point2d(b->x, b->y), color, 2);
        }
    }
}

//Vect - geometry vector 2D
class Vect {
    public:
    Blob begin, end;
    bool IsLeft(Blob);
    Vect(): begin(0,0), end(0,0) {}
    Vect(int _x1, int _y1, int _x2, int _y2): begin(_x1, _y1), end(_x2, _y2) {}
    Vect(Blob _begin, Blob _end): begin(_begin), end(_end) {}
    Vect(const Vect &v): begin(v.begin), end(v.end) {}
    string String() {
        char str[25];
        sprintf(str,"Vect:(%d;%d)->(%d;%d)", this->begin.x, this->begin.y, this->end.x, this->end.y);
        return string(str);
    }
};

//IsLeft - returns true if point is on the left
bool Vect::IsLeft(Blob p) {
    return ((this->begin.x-this->end.x)*(p.y-this->end.y) - (this->begin.y-this->end.y)*(p.x-this->end.x)) > 0;
}

//CentrTracker - tracks all objects on the scene
class CentrTracker{
    private:
    void Register(std::list<Blob>);
    void AllDissapear();
    public:
    int maxNoMatch;
    int maxBlobs;
    float tresholdDist;
    std::list<Track> Tracks;
    void Update(std::list<Blob>);
    void DrawCV(Mat &i);
    CentrTracker() {
        maxNoMatch = 15;
        maxBlobs = 10;
        tresholdDist = 15;
    }
    CentrTracker(int nomatch,int maxblobs,float dist): maxNoMatch(nomatch), maxBlobs(maxblobs), tresholdDist(dist){}
};

void CentrTracker::AllDissapear() {
    for (auto trit = this->Tracks.begin(); trit!= this->Tracks.end(); ++trit) {
        trit->nomatch++;
        trit->continued=false;
        if (trit->nomatch>this->maxNoMatch) {
            trit = this->Tracks.erase(trit);
        }
    }
}

void CentrTracker::Register(std::list<Blob> blst){
    for(auto b: blst){
        if (!b.exist) {
            this->Tracks.push_back(Track(b, this->maxBlobs));
        }
    }
}

void CentrTracker::Update(std::list<Blob> blst) {
    if (blst.size() == 0) {
        this->AllDissapear();
        return;
    }
    if (this->Tracks.size()==0) {
        this->Register(blst);
        return;
    }
    for (auto& b: blst) {
        auto minDist = this->tresholdDist;
        auto closestTr = this->Tracks.end();
        for(auto trit = this->Tracks.begin();trit!=this->Tracks.end(); trit++){
            auto dist = Distance(trit->Blobs[0], b);
            if (b.name != trit->Blobs[0].name) 
                continue;
            if (dist<minDist) {
                closestTr = trit;
                minDist = dist;
            }
        }
        if (closestTr != this->Tracks.end()) {
            closestTr->AddBlob(b);
            closestTr->nomatch = 0;
            b.exist = true;
        } 
    }

    for(auto tr = this->Tracks.begin();tr!=this->Tracks.end();) {
        if (!tr->continued) {
            // cout<<tr->ToJSON()<<endl;
            tr->nomatch++;
            if (tr->nomatch>this->maxNoMatch) {
                // cout<<"erase"<<endl;
                tr = this->Tracks.erase(tr);
            }
        }
        tr->continued=false;
        tr++;
    }
    this->Register(blst);
}

void CentrTracker::DrawCV(Mat &img) {
    for (auto t: this->Tracks){
        t.DrawCV(img);
    }
}
