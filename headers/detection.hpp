#include <opencv2/opencv.hpp>

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
    Detection(): classId(0),  confidence(0.), bbox(Rect(0,0,0,0)) {};
};

namespace std {
template<>
struct hash<Detection>{
    std::size_t operator()(const Detection& d) const{
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
    // point to the left of the line
    inline bool pointToTheLeft(Point p) {
        return ((this->beg.x-this->end.x)*(p.y-this->end.y) - (this->beg.y-this->end.y)*(p.x-this->end.x)) > 0;
    }
    Point ToVec() {
        return Point(end-beg); 
    }
    inline double cross(Point v1,Point v2){
        return v1.x*v2.y - v1.y*v2.x;
    }
    inline bool intersect(Line l) {
        return (cross(this->ToVec(),l.ToVec()) != 0); 
    }
    public: 
    cv::Point2i beg, end;
    Line() : beg{0,0}, end{0,0} {};
    Line(int x1, int y1, int x2, int y2) : beg{x1, y1}, end{x2, y2} {};
    Line(Point p1, Point p2) : beg(p1), end(p2) {};
    Line(const Line &l) : beg(l.beg), end(l.end) {}; 
    void Print() {
        std::cout << beg<< "->"<< end << std::endl;
    }
    void DrawCV(cv::Mat& img) {
        cv::line(img, beg, end, CV_RGB(128, 128, 128), 2);
        cv::circle(img, beg, 2, CV_RGB(128, 128, 128));
    }
    bool CrossedInDirection(Line l) {
        return (this->intersect(l) && l.pointToTheLeft(this->end) );
    }
};
