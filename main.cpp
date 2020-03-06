
#include <iostream>
#include <memory>
#include <string>
#include <cstdlib> // для system
#include "opencv2/opencv.hpp"
#include "yolo_grpc.grpc.pb.h"
#include <grpcpp/grpcpp.h>
#include <grpc/support/log.h>
using namespace std;
using grpc::Channel;
//using grpc::ClientAsyncResponseReader;
using grpc::ClientContext;
//using grpc::CompletionQueue;
using grpc::Status;
using yolo_grpc::STYolo;
class STYoloClient {
 public:
  explicit STYoloClient(std::shared_ptr<Channel> channel)
      : stub_(STYolo::NewStub(channel)) {}

  // Assembles the client's payload, sends it and presents the response back
  // from the server.
  std::string EndDetection(const std::string& cam_id) {
    // Data we are sending to the server.
    yolo_grpc::CamInfo request;
    request.set_cam_id(cam_id);

    // Container for the data we expect from the server.
    yolo_grpc::Response reply;

    // Context for the client. It could be used to convey extra information to
    // the server and/or tweak certain RPC behaviors.
    ClientContext context;

    // The producer-consumer queue we use to communicate asynchronously with the
    // gRPC runtime.
    //CompletionQueue cq;

    // Storage for the status of the RPC upon completion.
    Status status;

    // stub_->PrepareAsyncSayHello() creates an RPC object, returning
    // an instance to store in "call" but does not actually start the RPC
    // Because we are using the asynchronous API, we need to hold on to
    // the "call" instance in order to get updates on the ongoing RPC.
    /*
    std::unique_ptr<ClientAsyncResponseReader<yolo_grpc::Response> > rpc(
        stub_->PrepareAsyncEndDetection(&context, request, &cq));
    */
    status = stub_->EndDetection(&context, request, &reply);
    // StartCall initiates the RPC call
    //rpc->StartCall();

    // Request that, upon completion of the RPC, "reply" be updated with the
    // server's response; "status" with the indication of whether the operation
    // was successful. Tag the request with the integer 1.
    //rpc->Finish(&reply, &status, (void*)1);
    void* got_tag;
    bool ok = false;
    // Block until the next result is available in the completion queue "cq".
    // The return value of Next should always be checked. This return value
    // tells us whether there is any kind of event or the cq_ is shutting down.
    //GPR_ASSERT(cq.Next(&got_tag, &ok));

    // Verify that the result from "cq" corresponds, by its tag, our previous
    // request.
    ///GPR_ASSERT(got_tag == (void*)1);
    // ... and that the request was completed successfully. Note that "ok"
    // corresponds solely to the request for updates introduced by Finish().
    //GPR_ASSERT(ok);

    // Act upon the status of the actual RPC.
    if (status.ok()) {
      return reply.message();
    } else {
      return "RPC failed";
    }
  }

 private:
  // Out of the passed in Channel comes the stub, stored here, our view of the
  // server's exposed services.
  std::unique_ptr<STYolo::Stub> stub_;
};


int main() 
{ 
    string s = ".png";
    cv::Mat im = cv::imread("./data/car1.png");
    cv::imshow("[eq",im);
    vector<uchar> buf;
    cv::imencode(s,im,buf);
    cv::waitKey(0);
    cout << "Hello, world!" << endl;
    STYoloClient greeter(grpc::CreateChannel(
      "localhost:50051", grpc::InsecureChannelCredentials()));
    std::string user("world");
    std::string reply = greeter.EndDetection("0");  // The actual RPC call!
    std::cout << "Greeter received: " << reply << std::endl;
    return 0; 
}