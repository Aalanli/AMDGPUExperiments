
#include <cstdlib>

#include <ck/ck.hpp>
#include <ck/stream_config.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp>
#include <ck/library/tensor_operation_instance/add_device_operation_instance.hpp>

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;

// Compilation parameters for a[m, k] * b[k, n] = c[m, n]
using DeviceGemm =
                                      // clang-format off
                                      //  ########| AData| BData| CData| AccData| ALayout| BLayout| CLayout|           A|           B|           C|           GEMM| Block|  MPer|  NPer| K0Per| K1|      M1Per|      N1Per|   KPer|  M11N11Thread|  M11N11Thread|     ABlockTransfer|       ABlockTransfer| ABlockTransfer| ABlockTransfer|      ABlockTransfer|     ABlockTransfer|       ABlockTransfer|     BBlockTransfer|       BBlockTransfer| BBlockTransfer| BBlockTransfer|      BBlockTransfer|     BBlockTransfer|       BBlockTransfer|     CThreadTransfer|  CThreadTransfer|    CThreadTransfer|
                                      //  ########|  Type|  Type|  Type|    Type|        |        |        | Elementwise| Elementwise| Elementwise| Specialization|  Size| Block| Block| Block|   | ThreadM111| ThreadN111| Thread| ClusterM110Xs| ClusterN110Xs| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor| ThreadSliceLengths| ThreadClusterLengths|  ThreadCluster|      SrcAccess|     SrcVectorTensor|    SrcVectorTensor|      DstVectorTensor|        SrcDstAccess|  SrcDstVectorDim| DstScalarPerVector|
                                      //  ########|      |      |      |        |        |        |        |   Operation|   Operation|   Operation|               |      |      |      |      |   |           |           |       |              |              |        K0_M0_M1_K1|          K0_M0_M1_K1|   ArrangeOrder|          Order| Lengths_K0_M0_M1_K1| ContiguousDimOrder|  Lengths_K0_M0_M1_K1|        K0_N0_N1_K1|          K0_N0_N1_K1|   ArrangeOrder|          Order| Lengths_K0_N0_N1_K1| ContiguousDimOrder|  Lengths_K0_N0_N1_K1|               Order|                 |                   |
                                      //  ########|      |      |      |        |        |        |        |            |            |            |               |      |      |      |      |   |           |           |       |              |              |                   |                     |               |               |                    |                   |                     |                   |                     |               |               |                    |                   |                     |                    |                 |                   |
        ck::tensor_operation::device::DeviceGemmDl<   F32,   F32,   F32,     F32,     Row,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDefault,   256,   128,   128,    16,  1,          4,          4,      1,       S<8, 2>,       S<8, 2>,      S<8, 1, 1, 1>,      S<2, 1, 128, 1>,  S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<4, 1, 1, 1>,      S<1, 2, 0, 3>,        S<1, 1, 1, 1>,      S<2, 1, 4, 1>,      S<8, 1,  32, 1>,  S<0, 3, 1, 2>,  S<0, 3, 1, 2>,       S<1, 1, 4, 1>,      S<0, 3, 1, 2>,        S<1, 1, 4, 1>, S<0, 1, 2, 3, 4, 5>,                5,                  4>;
        // clang-format on

const auto gemm = DeviceGemm{};
const auto elementwise_op = PassThrough{};

extern "C" __attribute__((visibility("default"))) bool ck_gemm_f32(float* a, float* b, float* c, int m, int k, int n) {
    auto invoker = gemm.MakeInvoker();
    auto argument = gemm.MakeArgument(
        a, b, c,
        m,
        n,
        k,
        k,
        n,
        n,
        elementwise_op,
        elementwise_op,
        elementwise_op);
    
    invoker.Run(argument, StreamConfig{nullptr, false});

    return true;
}