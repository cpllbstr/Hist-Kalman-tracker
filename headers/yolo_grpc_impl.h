#pragma once
#include "yolo_grpc.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
#include <opencv2/core.hpp>
#include <detection.hpp>
#include <track.hpp>

using namespace std;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using yolo_grpc::STYolo;

class STYoloClient {
public:
    explicit STYoloClient(std::shared_ptr<Channel> channel)
        : stub_(STYolo::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    std::string EndDetection(const string cam_id, Track &tr, cv::Mat &im) {
        auto request = new yolo_grpc::CamInfo();
        vector<uchar> buf;
        cv::imencode(".png", im ,buf);
        string strdata(buf.begin(), buf.end());
        request->set_cam_id(cam_id);
        request->set_image(move(strdata));
        //string s(reinterpret_cast<char*>(buf.data()));
        //request.set_allocated_image(&s);
        yolo_grpc::Detection (det);
        det.set_x_left(tr.prev_det.bbox.x);
        det.set_y_top(tr.prev_det.bbox.y);
        det.set_width(tr.prev_det.bbox.width);
        det.set_height(tr.prev_det.bbox.height);
        det.set_line_id("0");
        request->set_allocated_detection(&det);

        
        yolo_grpc::Response reply;
        ClientContext context;
        Status status = stub_->EndDetection(&context, *request, &reply);

        // StartCall initiates the RPC call
        if (status.ok()) {
            return reply.message() + "\n" + reply.warning();
        } else {
            std::cout << status.error_code() << ": " << status.error_message()
                      << std::endl;
            return "RPC failed";
        }
    }

private:
    // Out of the passed in Channel comes the stub, stored here, our view of the
    // server's exposed services.
    std::unique_ptr<STYolo::Stub> stub_;
};