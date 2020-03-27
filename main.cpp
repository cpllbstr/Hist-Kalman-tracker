// This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

// Usage example:  ./object_detection_yolo.out --video=run.mp4
//                 ./object_detection_yolo.out --image=bird.jpg
#include <fstream>
#include <sstream>
#include <iostream>
#include <kalman_tracker.hpp>
#include <toml.hpp>
#include <opencv2/opencv.hpp>

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./object_detection_yolo.out --image=dog.jpg \n\t\t./object_detection_yolo.out --video=run_sm.mp4}"
"{image i        |<none>| input image   }"
"{video v       |<none>| input video   }"
"{device d      |<none>| input device  }"
;
using namespace cv;
using namespace dnn;
using namespace std;

// Initialize the parameters
float confThreshold = 0.5; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;  // Width of network's input image
int inpHeight = 416; // Height of network's input image
string 
    yolo_cfg, 
    yolo_weights;
int 
    // distT,
    camera_id;
    // pointsC,
    // nomatch;
// float histT;

vector<string> classes;



// Needed only for debug
void drawDets(list<Detection> dets, Mat& frame) {
    //Draw a rectangle displaying the bounding box
    for (auto det: dets) {
        rectangle(frame,det.bbox, Scalar(255, 178, 50), 3);

        //Get the label for the class name and its confidence
        string label = format("%.2f", det.confidence);
        if (!classes.empty()) {
            CV_Assert(det.classId < (int)classes.size());
            label = classes[det.classId] + ":" + label;
        }

        //Display the label at the top of the bounding box
        int baseLine;
        Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        // top = max(top, labelSize.height);
        // rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
        putText(frame, label, det.bbox.tl(), FONT_HERSHEY_SIMPLEX, 0.75, CV_RGB(250,230,0),1.5);
    }
}


// Get the names of the output layers
vector<String> getOutputsNames(const Net& net);

// Remove the bounding boxes with low confidence using non-maxima suppression
list<Detection> postprocess(Mat& frame, const vector<Mat>& outs) {
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    
    for (size_t i = 0; i < outs.size(); ++i) {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
            Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > confThreshold) {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                left = left < 0 ? 0 : left;
                top =  top < 0 ? 0 : top;
                left = left > frame.cols ? frame.cols : left;
                top = top > frame.rows ? frame.rows : top;
                if (top+height>frame.rows) height = frame.rows-top;
                if (left+width>frame.cols) width = frame.cols-left;
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(Rect(left, top, width, height));
            }
        }
    }
    
    // Perform non maximum suppression to eliminate redundant overlapping boxes with
    // lower confidences
    list<Detection> res;
    vector<int> indices;
    NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
    for(auto ind : indices) {
        Detection d;
        d.classId = classIds[ind];
        d.confidence = confidences[ind];
        d.bbox =  boxes[ind];
        res.push_back(d);
    }
    return res;
}


void process_camera(int cam_id, string modelConfiguration , string modelWeights) {
// Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // String modelConfiguration = "yolov3.cfg";
    // String modelWeights = "yolov3.weights";


    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);
    
    // Open a video file or an image file or a camera stream.
    string  outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;

    cap.open(cam_id);
    if(!cap.isOpened()){
        cout << "Cannot open camera with id: "<<cam_id<<endl;
        return;
    } 
    KalmanTracker ktr;
    // Process frames.
    
    // ktr.LoadConfig("./config.toml");
    cout << ktr.histTreshold << ktr.maxNoMatch << ktr.maxPointsCount << ktr.tresholdDist << endl;
    
    auto start_s = std::chrono::steady_clock::now();
    auto time = getTickCount();
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }
        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        auto start = std::chrono::steady_clock::now();
        //Sets the input to the network
        net.setInput(blob);
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        auto duration = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        // cout << "Time on forwarding: " << duration.count() << endl;
        // Remove the bounding boxes with low confidence
        auto dets = postprocess(frame, outs);
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        auto now = getTickCount();
        double dtime =  (now - time)/getTickFrequency();
        time = now;
        // cout <<"TIME: "<<dtime<<endl;
        auto dduration = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        // cout << format("dduration: %d ms\n", dduration.count());
        ktr.Update(dets, frame,  1.0/* dtime */);
        string label = format("Inference time for a frame : %.2f ms|| overall duration %d", t, dduration.count());
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        
        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        // if (parser.has("image")) imwrite(outputFile, detectedFrame);
        drawDets(dets, detectedFrame);
        ktr.DrawCV(detectedFrame);
        
        video.write(detectedFrame);
        imshow("test", detectedFrame);
    }
    cap.release();
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
    static vector<String> names;
    if (names.empty())
    {
        //Get the indices of the output layers, i.e. the layers with unconnected outputs
        vector<int> outLayers = net.getUnconnectedOutLayers();
        
        //get the names of all the layers in the network
        vector<String> layersNames = net.getLayerNames();
        
        // Get the names of the output layers in names
        names.resize(outLayers.size());
        for (size_t i = 0; i < outLayers.size(); ++i)
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

void process_video(string vid, string modelConfiguration , string modelWeights) {
// Load names of classes
    string classesFile = "coco.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);

    // String modelConfiguration = "yolov3.cfg";
    // String modelWeights = "yolov3.weights";


    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableBackend(DNN_BACKEND_CUDA);
    net.setPreferableTarget(DNN_TARGET_CUDA);
    
    // Open a video file or an image file or a camera stream.
    string  outputFile;
    VideoCapture cap;
    VideoWriter video;
    Mat frame, blob;

    ifstream ifile(vid);
    cap.open(vid);
    if(!cap.isOpened()){
        cout << "Cannot open: "<<vid<<endl;
        return;
    } 
    vid.replace(vid.end()-4, vid.end(), "_out.avi");
    outputFile = vid;
    video.open(outputFile, VideoWriter::fourcc('M','J','P','G'), 28, Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));
    KalmanTracker ktr;
    // Process frames.
    
    // ktr.LoadConfig("./config.toml");
    cout << ktr.histTreshold << ktr.maxNoMatch << ktr.maxPointsCount << ktr.tresholdDist << endl;
    
    auto start_s = std::chrono::steady_clock::now();
    auto time = getTickCount();
    while (waitKey(1) < 0)
    {
        // get frame from the video
        cap >> frame;

        // Stop the program if reached end of video
        if (frame.empty()) {
            cout << "Done processing !!!" << endl;
            cout << "Output file is stored as " << outputFile << endl;
            waitKey(3000);
            break;
        }
        // Create a 4D blob from a frame.
        blobFromImage(frame, blob, 1/255.0, cv::Size(inpWidth, inpHeight), Scalar(0,0,0), true, false);
        auto start = std::chrono::steady_clock::now();
        //Sets the input to the network
        net.setInput(blob);
        // Runs the forward pass to get output of the output layers
        vector<Mat> outs;
        net.forward(outs, getOutputsNames(net));
        
        auto duration = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        // cout << "Time on forwarding: " << duration.count() << endl;
        // Remove the bounding boxes with low confidence
        auto dets = postprocess(frame, outs);
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        vector<double> layersTimes;
        double freq = getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        auto now = getTickCount();
        double dtime =  (now - time)/getTickFrequency();
        time = now;
        // cout <<"TIME: "<<dtime<<endl;
        auto dduration = std::chrono::duration_cast<chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        // cout << format("dduration: %d ms\n", dduration.count());
        ktr.Update(dets, frame,  1.0/* dtime */);
        string label = format("Inference time for a frame : %.2f ms|| overall duration %d", t, dduration.count());
        putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
        
        // Write the frame with the detection boxes
        Mat detectedFrame;
        frame.convertTo(detectedFrame, CV_8U);
        // if (parser.has("image")) imwrite(outputFile, detectedFrame);
        drawDets(dets, detectedFrame);
        ktr.DrawCV(detectedFrame);
        
        video.write(detectedFrame);
        imshow("test", detectedFrame);
    }
    cap.release();
}

int main(int argc, char** argv)
{
    string config_s = "./config.toml";
    auto config=toml::parse(config_s);
    //net cofiguration
    try {
        yolo_cfg = config["net"]["cfg"].as_string();
        yolo_weights = config["net"]["weights"].as_string();
        confThreshold = config["net"]["confThreshold"].as_floating();
        nmsThreshold = config["net"]["nmsThreshold"].as_floating();
        inpWidth = config["net"]["inpWidth"].as_integer(); 
        inpHeight = config["net"]["inpHeight"].as_integer();
    } catch(const exception &e) {
        cout << "Cannot parse data from "<<config_s <<" [net]! Check your syntax: ";
        cout << e.what() << endl;
        return -1;
    }
    //input cofiguration
    try {
        if (config.at("input").contains("video")) {
            auto vid_path = config["input"]["video"].as_string();
            try {
                process_video(vid_path, yolo_cfg, yolo_weights);
            } catch (const exception &e) {
                cout << e.what() << endl;
                return -1;
            }
        } else if (config.at("input").contains("camera_id")) {
            auto cam_id = config["input"]["camera_id"].as_integer();
             try {
                process_camera(cam_id, yolo_cfg, yolo_weights);
            } catch (const exception &e) {
                cout << e.what() << endl;
                return -1;
            }
        }
    } catch(const exception &e) {
        cout << "Cannot parse data from "<<config_s <<" [input]! Check your syntax: ";
        cout << e.what() << endl;
        return -1;
    }
}
