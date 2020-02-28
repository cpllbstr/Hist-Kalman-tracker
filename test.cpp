#include <kalman_tracker.hpp>


Mat calcHistRGB(Mat img) {
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

/* void testHist(){
    // auto res = compareHist(hist1, hist2, HISTCMP_BHATTACHARYYA); 
    // cout << res << endl;
    
    // auto img2 = imread("/home/lbstr/Downloads/Telegram Desktop/1.jpg");
    auto img1 = imread("./car1.png");
    auto img2 = imread("./test.png");
    auto img3 = imread("./car1_1.png");


    // imshow("img1", img1);
    // imshow("imgq1", eqim1);
    // waitKey(0);

    auto hist1 = calcHistRGB(img1);
    auto hist2 = calcHistRGB(img2);
    auto hist3 = calcHistRGB(img3);

    cout <<"Same img cor: "<< compareHist(hist1, hist1, HISTCMP_CORREL) << endl;
    cout <<"Diff car cor: " << compareHist(hist2, hist1, HISTCMP_CORREL) << endl;
    cout <<"Same car cor: " << compareHist(hist3, hist1, HISTCMP_CORREL) << endl<<endl;

    cout <<"Same img bar: "<< compareHist(hist1, hist1, HISTCMP_BHATTACHARYYA) << endl;
    cout <<"Diff car bar: " << compareHist(hist2, hist1, HISTCMP_BHATTACHARYYA) << endl;
    cout <<"Same car bar: " << compareHist(hist3, hist1, HISTCMP_BHATTACHARYYA) << endl;    

    // imshow("h2", img(Rect(300, 300, 300, 300)));
    // auto res = compareHist(hist1, hist2, HISTCMP_BHATTACHARYYA); 
    // cout << res << endl;
} */

class Test {
    public:
    string str;
    void Delete() {
        this->~Test();
    }
    ~Test() {
    }
};

int main(int argc, char const *argv[]) {
    auto Green  = CV_RGB(0,255,0);
    auto Red = CV_RGB(255, 0,0);
    Mat G  = Mat::zeros(100, 100, CV_32FC3);
    G.setTo(Green);
    Mat R  = Mat::zeros(100, 100, CV_32FC3);
    R.setTo(Red);
    Mat img  = Mat::zeros(1024, 1024, CV_8UC3);
    auto detect = []() {return tuple<int, int>(10 + sin(getTickCount()), 10 + cos(getTickCount()));};
    Detection det;
    det.bbox = Rect(0,0,10,10);
    KalmanTracker ktr;
    for (int i =0; i<10; i++){
        auto[dx, dy] = detect();
        ktr.Update({det}, img, 1);
        det.bbox.x+=dx;
        det.bbox.y+=dy;
        cout << det.bbox << endl;
    } 
    for(auto p: ktr.Tracks.front().Points) {
        cout << p << endl; 
    }
    cout << ktr.Tracks.size() << endl;
    ktr.DrawCV(img);
    imshow("im",img);
    waitKey(0);

}
