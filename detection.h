#include <opencv2/opencv.hpp>

using namespace cv;

struct Detection {
    public:
    int classId;
    float confidence;
    Rect bbox;
    bool registered =  false;
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
      ((hash<int>()(d.bbox.x)) ^ (hash<int>()(d.bbox.y)) ^ 
      (hash<int>()(d.bbox.height)) ^ (hash<int>()(d.bbox.width))) >> 1);
    }
};
}

