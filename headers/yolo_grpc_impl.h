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


struct ServerNotRespondingException : public exception {
    const char* what() const throw() {
        return "Server is not responding. Check your enviroment variable DETECTOR_ADDR_PORT or if server is running";
    }
};


class STYoloClient {
public:
    explicit STYoloClient(std::shared_ptr<Channel> channel) : stub_(STYolo::NewStub(channel)) {}

    // Assembles the client's payload, sends it and presents the response back
    // from the server.
    std::string SendDetection(const string cam_id, int line_id, Track &tr, cv::Mat &im) {
        auto request = new yolo_grpc::CamInfo();
        vector<uchar> buf;
        cv::imencode(".png", im,buf);
        string strdata(buf.begin(), buf.end());
        request->set_cam_id(cam_id);
        request->set_image(move(strdata));
        request->set_timestamp(time(0));

        for (auto &pnt:tr.Points) {
            auto pnt_tosend = request->add_track();
            pnt_tosend->set_x(pnt.x);
            pnt_tosend->set_y(pnt.y);
        }
        //string s(reinterpret_cast<char*>(buf.data()));
        //request.set_allocated_image(&s);
        yolo_grpc::Detection (det);
        det.set_x_left(tr.prev_det.bbox.x);
        det.set_y_top(tr.prev_det.bbox.y);
        det.set_width(tr.prev_det.bbox.width);
        det.set_height(tr.prev_det.bbox.height);
        det.set_line_id(line_id);


        request->set_allocated_detection(&det);


        yolo_grpc::Response reply;
        ClientContext context;
        Status status = stub_->SendDetection(&context, *request, &reply);

        // StartCall initiates the RPC call
        if (status.ok()) {
            return reply.message() + "\n" + reply.warning();
        } else {
            std::cout << status.error_code() << ": " << status.error_message()
                      << std::endl;
            return "RPC failed";
        }
    }

    void ConfigUpdater() {
        using namespace yolo_grpc;
        using namespace grpc;
        cout << "Subscrubing to configuration updates\n";
        
        ClientContext context;
        // context.set_deadline(grpc::TimePoint<int>(1000));
        auto stream  = stub_->ConfigUpdater(&context);
        // shared_ptr<ClientReaderWriter<Response, Config>> stream(move(stub_->ConfigUpdater(&context).release()));
        auto resp = new Response();
        resp->set_allocated_message(new string("Subscribe Response"));
        resp->Clear();
        Config newconf;
        if (stream->Write(*resp)) {
            stream->Read(&newconf);
            cout << "Got UID:" << newconf.uid() << endl;
        } else {
            cout << "Cannot locate server" << endl;
            throw ServerNotRespondingException();
        }

        for(;;) {
            Config newconf;
            if (!stream->Read(&newconf)) {
                continue;
            } else {
                auto resp = new Response();
                // resp->set_allocated_message(new string("Response!"))
                resp->set_allocated_message(new string("New conf loaded"));
                if (!stream->Write(*resp)) {
                    cout << "Error stream is not responding!" << endl;
                }
            }
        }
    }

private:
    // Out of the passed in Channel comes the stub, stored here, our view of the
    // server's exposed services.
    std::unique_ptr<STYolo::Stub> stub_;
};