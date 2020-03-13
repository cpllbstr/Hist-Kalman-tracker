// Generated by the gRPC C++ plugin.
// If you make any local change, they will be lost.
// source: yolo_grpc.proto
#ifndef GRPC_yolo_5fgrpc_2eproto__INCLUDED
#define GRPC_yolo_5fgrpc_2eproto__INCLUDED

#include "yolo_grpc.pb.h"

#include <functional>
#include <grpc/impl/codegen/port_platform.h>
#include <grpcpp/impl/codegen/async_generic_service.h>
#include <grpcpp/impl/codegen/async_stream.h>
#include <grpcpp/impl/codegen/async_unary_call.h>
#include <grpcpp/impl/codegen/client_callback.h>
#include <grpcpp/impl/codegen/client_context.h>
#include <grpcpp/impl/codegen/completion_queue.h>
#include <grpcpp/impl/codegen/message_allocator.h>
#include <grpcpp/impl/codegen/method_handler.h>
#include <grpcpp/impl/codegen/proto_utils.h>
#include <grpcpp/impl/codegen/rpc_method.h>
#include <grpcpp/impl/codegen/server_callback.h>
#include <grpcpp/impl/codegen/server_callback_handlers.h>
#include <grpcpp/impl/codegen/server_context.h>
#include <grpcpp/impl/codegen/service_type.h>
#include <grpcpp/impl/codegen/status.h>
#include <grpcpp/impl/codegen/stub_options.h>
#include <grpcpp/impl/codegen/sync_stream.h>

namespace yolo_grpc {

// STYolo - service for second and third yolo stages 
// in car number search cascade
class STYolo final {
 public:
  static constexpr char const* service_full_name() {
    return "yolo_grpc.STYolo";
  }
  class StubInterface {
   public:
    virtual ~StubInterface() {}
    virtual ::grpc::Status EndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::yolo_grpc::Response* response) = 0;
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::yolo_grpc::Response>> AsyncEndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::yolo_grpc::Response>>(AsyncEndDetectionRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::yolo_grpc::Response>> PrepareAsyncEndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReaderInterface< ::yolo_grpc::Response>>(PrepareAsyncEndDetectionRaw(context, request, cq));
    }
    class experimental_async_interface {
     public:
      virtual ~experimental_async_interface() {}
      virtual void EndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo* request, ::yolo_grpc::Response* response, std::function<void(::grpc::Status)>) = 0;
      virtual void EndDetection(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::yolo_grpc::Response* response, std::function<void(::grpc::Status)>) = 0;
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      virtual void EndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo* request, ::yolo_grpc::Response* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      #else
      virtual void EndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo* request, ::yolo_grpc::Response* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      #endif
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      virtual void EndDetection(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::yolo_grpc::Response* response, ::grpc::ClientUnaryReactor* reactor) = 0;
      #else
      virtual void EndDetection(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::yolo_grpc::Response* response, ::grpc::experimental::ClientUnaryReactor* reactor) = 0;
      #endif
    };
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    typedef class experimental_async_interface async_interface;
    #endif
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    async_interface* async() { return experimental_async(); }
    #endif
    virtual class experimental_async_interface* experimental_async() { return nullptr; }
  private:
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::yolo_grpc::Response>* AsyncEndDetectionRaw(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::grpc::CompletionQueue* cq) = 0;
    virtual ::grpc::ClientAsyncResponseReaderInterface< ::yolo_grpc::Response>* PrepareAsyncEndDetectionRaw(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::grpc::CompletionQueue* cq) = 0;
  };
  class Stub final : public StubInterface {
   public:
    Stub(const std::shared_ptr< ::grpc::ChannelInterface>& channel);
    ::grpc::Status EndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::yolo_grpc::Response* response) override;
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::yolo_grpc::Response>> AsyncEndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::yolo_grpc::Response>>(AsyncEndDetectionRaw(context, request, cq));
    }
    std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::yolo_grpc::Response>> PrepareAsyncEndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::grpc::CompletionQueue* cq) {
      return std::unique_ptr< ::grpc::ClientAsyncResponseReader< ::yolo_grpc::Response>>(PrepareAsyncEndDetectionRaw(context, request, cq));
    }
    class experimental_async final :
      public StubInterface::experimental_async_interface {
     public:
      void EndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo* request, ::yolo_grpc::Response* response, std::function<void(::grpc::Status)>) override;
      void EndDetection(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::yolo_grpc::Response* response, std::function<void(::grpc::Status)>) override;
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      void EndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo* request, ::yolo_grpc::Response* response, ::grpc::ClientUnaryReactor* reactor) override;
      #else
      void EndDetection(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo* request, ::yolo_grpc::Response* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      #endif
      #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      void EndDetection(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::yolo_grpc::Response* response, ::grpc::ClientUnaryReactor* reactor) override;
      #else
      void EndDetection(::grpc::ClientContext* context, const ::grpc::ByteBuffer* request, ::yolo_grpc::Response* response, ::grpc::experimental::ClientUnaryReactor* reactor) override;
      #endif
     private:
      friend class Stub;
      explicit experimental_async(Stub* stub): stub_(stub) { }
      Stub* stub() { return stub_; }
      Stub* stub_;
    };
    class experimental_async_interface* experimental_async() override { return &async_stub_; }

   private:
    std::shared_ptr< ::grpc::ChannelInterface> channel_;
    class experimental_async async_stub_{this};
    ::grpc::ClientAsyncResponseReader< ::yolo_grpc::Response>* AsyncEndDetectionRaw(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::grpc::CompletionQueue* cq) override;
    ::grpc::ClientAsyncResponseReader< ::yolo_grpc::Response>* PrepareAsyncEndDetectionRaw(::grpc::ClientContext* context, const ::yolo_grpc::CamInfo& request, ::grpc::CompletionQueue* cq) override;
    const ::grpc::internal::RpcMethod rpcmethod_EndDetection_;
  };
  static std::unique_ptr<Stub> NewStub(const std::shared_ptr< ::grpc::ChannelInterface>& channel, const ::grpc::StubOptions& options = ::grpc::StubOptions());

  class Service : public ::grpc::Service {
   public:
    Service();
    virtual ~Service();
    virtual ::grpc::Status EndDetection(::grpc::ServerContext* context, const ::yolo_grpc::CamInfo* request, ::yolo_grpc::Response* response);
  };
  template <class BaseClass>
  class WithAsyncMethod_EndDetection : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithAsyncMethod_EndDetection() {
      ::grpc::Service::MarkMethodAsync(0);
    }
    ~WithAsyncMethod_EndDetection() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status EndDetection(::grpc::ServerContext* /*context*/, const ::yolo_grpc::CamInfo* /*request*/, ::yolo_grpc::Response* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestEndDetection(::grpc::ServerContext* context, ::yolo_grpc::CamInfo* request, ::grpc::ServerAsyncResponseWriter< ::yolo_grpc::Response>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  typedef WithAsyncMethod_EndDetection<Service > AsyncService;
  template <class BaseClass>
  class ExperimentalWithCallbackMethod_EndDetection : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    ExperimentalWithCallbackMethod_EndDetection() {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::Service::
    #else
      ::grpc::Service::experimental().
    #endif
        MarkMethodCallback(0,
          new ::grpc_impl::internal::CallbackUnaryHandler< ::yolo_grpc::CamInfo, ::yolo_grpc::Response>(
            [this](
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
                   ::grpc::CallbackServerContext*
    #else
                   ::grpc::experimental::CallbackServerContext*
    #endif
                     context, const ::yolo_grpc::CamInfo* request, ::yolo_grpc::Response* response) { return this->EndDetection(context, request, response); }));}
    void SetMessageAllocatorFor_EndDetection(
        ::grpc::experimental::MessageAllocator< ::yolo_grpc::CamInfo, ::yolo_grpc::Response>* allocator) {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::GetHandler(0);
    #else
      ::grpc::internal::MethodHandler* const handler = ::grpc::Service::experimental().GetHandler(0);
    #endif
      static_cast<::grpc_impl::internal::CallbackUnaryHandler< ::yolo_grpc::CamInfo, ::yolo_grpc::Response>*>(handler)
              ->SetMessageAllocator(allocator);
    }
    ~ExperimentalWithCallbackMethod_EndDetection() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status EndDetection(::grpc::ServerContext* /*context*/, const ::yolo_grpc::CamInfo* /*request*/, ::yolo_grpc::Response* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    virtual ::grpc::ServerUnaryReactor* EndDetection(
      ::grpc::CallbackServerContext* /*context*/, const ::yolo_grpc::CamInfo* /*request*/, ::yolo_grpc::Response* /*response*/)
    #else
    virtual ::grpc::experimental::ServerUnaryReactor* EndDetection(
      ::grpc::experimental::CallbackServerContext* /*context*/, const ::yolo_grpc::CamInfo* /*request*/, ::yolo_grpc::Response* /*response*/)
    #endif
      { return nullptr; }
  };
  #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
  typedef ExperimentalWithCallbackMethod_EndDetection<Service > CallbackService;
  #endif

  typedef ExperimentalWithCallbackMethod_EndDetection<Service > ExperimentalCallbackService;
  template <class BaseClass>
  class WithGenericMethod_EndDetection : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithGenericMethod_EndDetection() {
      ::grpc::Service::MarkMethodGeneric(0);
    }
    ~WithGenericMethod_EndDetection() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status EndDetection(::grpc::ServerContext* /*context*/, const ::yolo_grpc::CamInfo* /*request*/, ::yolo_grpc::Response* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
  };
  template <class BaseClass>
  class WithRawMethod_EndDetection : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithRawMethod_EndDetection() {
      ::grpc::Service::MarkMethodRaw(0);
    }
    ~WithRawMethod_EndDetection() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status EndDetection(::grpc::ServerContext* /*context*/, const ::yolo_grpc::CamInfo* /*request*/, ::yolo_grpc::Response* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    void RequestEndDetection(::grpc::ServerContext* context, ::grpc::ByteBuffer* request, ::grpc::ServerAsyncResponseWriter< ::grpc::ByteBuffer>* response, ::grpc::CompletionQueue* new_call_cq, ::grpc::ServerCompletionQueue* notification_cq, void *tag) {
      ::grpc::Service::RequestAsyncUnary(0, context, request, response, new_call_cq, notification_cq, tag);
    }
  };
  template <class BaseClass>
  class ExperimentalWithRawCallbackMethod_EndDetection : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    ExperimentalWithRawCallbackMethod_EndDetection() {
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
      ::grpc::Service::
    #else
      ::grpc::Service::experimental().
    #endif
        MarkMethodRawCallback(0,
          new ::grpc_impl::internal::CallbackUnaryHandler< ::grpc::ByteBuffer, ::grpc::ByteBuffer>(
            [this](
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
                   ::grpc::CallbackServerContext*
    #else
                   ::grpc::experimental::CallbackServerContext*
    #endif
                     context, const ::grpc::ByteBuffer* request, ::grpc::ByteBuffer* response) { return this->EndDetection(context, request, response); }));
    }
    ~ExperimentalWithRawCallbackMethod_EndDetection() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable synchronous version of this method
    ::grpc::Status EndDetection(::grpc::ServerContext* /*context*/, const ::yolo_grpc::CamInfo* /*request*/, ::yolo_grpc::Response* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    #ifdef GRPC_CALLBACK_API_NONEXPERIMENTAL
    virtual ::grpc::ServerUnaryReactor* EndDetection(
      ::grpc::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)
    #else
    virtual ::grpc::experimental::ServerUnaryReactor* EndDetection(
      ::grpc::experimental::CallbackServerContext* /*context*/, const ::grpc::ByteBuffer* /*request*/, ::grpc::ByteBuffer* /*response*/)
    #endif
      { return nullptr; }
  };
  template <class BaseClass>
  class WithStreamedUnaryMethod_EndDetection : public BaseClass {
   private:
    void BaseClassMustBeDerivedFromService(const Service* /*service*/) {}
   public:
    WithStreamedUnaryMethod_EndDetection() {
      ::grpc::Service::MarkMethodStreamed(0,
        new ::grpc::internal::StreamedUnaryHandler< ::yolo_grpc::CamInfo, ::yolo_grpc::Response>(std::bind(&WithStreamedUnaryMethod_EndDetection<BaseClass>::StreamedEndDetection, this, std::placeholders::_1, std::placeholders::_2)));
    }
    ~WithStreamedUnaryMethod_EndDetection() override {
      BaseClassMustBeDerivedFromService(this);
    }
    // disable regular version of this method
    ::grpc::Status EndDetection(::grpc::ServerContext* /*context*/, const ::yolo_grpc::CamInfo* /*request*/, ::yolo_grpc::Response* /*response*/) override {
      abort();
      return ::grpc::Status(::grpc::StatusCode::UNIMPLEMENTED, "");
    }
    // replace default version of method with streamed unary
    virtual ::grpc::Status StreamedEndDetection(::grpc::ServerContext* context, ::grpc::ServerUnaryStreamer< ::yolo_grpc::CamInfo,::yolo_grpc::Response>* server_unary_streamer) = 0;
  };
  typedef WithStreamedUnaryMethod_EndDetection<Service > StreamedUnaryService;
  typedef Service SplitStreamedService;
  typedef WithStreamedUnaryMethod_EndDetection<Service > StreamedService;
};

}  // namespace yolo_grpc


#endif  // GRPC_yolo_5fgrpc_2eproto__INCLUDED