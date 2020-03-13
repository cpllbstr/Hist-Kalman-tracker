#pragma once
#include <opencv2/opencv.hpp>
#include <toml.hpp>

using namespace cv;

struct Detection {
public:
    int classId;
    float confidence;
    Rect bbox;
    bool appended =  false;
    Point2d get_center() {
        return Point(bbox.x+int(bbox.width/2), bbox.y+int(bbox.height/2));
    };
    bool operator==(const Detection &other) const {
        return
            this->classId == other.classId &&
            this->confidence == other.confidence &&
            this->bbox == other.bbox;
    };
    Detection(const Detection &d) : 
        classId(d.classId),
        confidence(d.confidence),
        bbox(d.bbox) 
        {};
    Detection(): classId(0),  confidence(0.), bbox(Rect(0,0,0,0)) {};
};

namespace std {
template<>
struct hash<Detection> {
    std::size_t operator()(const Detection& d) const {
        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:
        return (((hash<int>()(d.classId) ^ (hash<float>()(d.confidence) << 1)) >> 1) ^
                (((hash<int>()(d.bbox.x)) ^ (hash<int>()(d.bbox.y)) <<1) ^
                 (hash<int>()(d.bbox.height)) ^ (hash<int>()(d.bbox.width))) >> 1);
    }
};
}

class Line {
private:
    inline bool pointToTheRight(Point p) {
        return ((this->beg.x-this->end.x)*(p.y-this->end.y) - (this->beg.y-this->end.y)*(p.x-this->end.x)) > 0;
    }
    inline bool pointInRange(Point p) {
        auto offset=25;
        auto left = min(this->beg.x-offset, this->end.x-offset);
        auto top = min(this->beg.y-offset, this->end.y-offset);
        auto right = max(this->beg.x+offset, this->end.x+offset);
        auto bot = max(this->beg.y+offset, this->end.y+offset);
        return p.x >left and p.x<right and p.y>top and p.y<bot;
    }
    Point ToVec() {
        return  end-beg;
    }
    inline double cross(Point v1,Point v2) {
        return v1.x*v2.y - v1.y*v2.x;
    }
    inline bool intersect(Line detl) {
        using namespace std;
        auto sign = [](float x) {
            if (x>0) return 1;
            if (x<0) return -1;
            else return 0;
        };
        auto v1 = detl.beg - this->beg;
        auto v2 = detl.end -this->beg;
        auto vecl = this->ToVec();
        return  (sign(v1.x*vecl.y-vecl.x*v1.y) != sign(v2.x*vecl.y-vecl.x*v2.y));
    }
public:
    int id;
    cv::Point2i beg, end;
    Line() : beg{0,0}, end{0,0}, id(0) {};
    Line(int x1, int y1, int x2, int y2) : beg{x1, y1}, end{x2, y2} {};
    Line(Point p1, Point p2) : beg(p1), end(p2), id(-1) {};
    Line(const Line &l) : beg(l.beg), end(l.end), id(l.id) {};
    Line(const Line &&l) : beg(l.beg), end(l.end), id(l.id) {};
    Line(toml::table);
    void Print() {
        std::cout << beg<< "->"<< end << std::endl;
    }
    void DrawCV(cv::Mat& img) {
        // @TODO: draw direction of detection line
        cv::line(img, beg, end, CV_RGB(128, 0, 255), 2);
        putText(img, "DL: "+std::to_string(id),end,FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0,230,0),1.5);
        cv::circle(img, end, 4, CV_RGB(128, 0, 128), 2);
    }
    //returns true if lines crossed and this line's second point is to the right of detection line
    bool CrossedInDirection(Line detection_line) {
        return \
               detection_line.pointInRange(this->beg) && \
               this->intersect(detection_line) && \
               detection_line.intersect(*this) && \
               detection_line.pointToTheRight(this->end);
    }
};

Line::Line(toml::table detline) {
    auto p1 = detline["beg"].as_array();
    auto p2 = detline["end"].as_array();
    this->beg = Point2i(p1[0].as_integer(), p1[1].as_integer());
    this->end = Point2i(p2[0].as_integer(), p2[1].as_integer());
    this->id = detline["id"].as_integer();
    std::cout << this->id << std::endl;
}