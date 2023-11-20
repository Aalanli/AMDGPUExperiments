
#include <ck/stream_config.hpp>
#include <cstring>
#include <iomanip>
#include <vector>
#include <iostream>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"

#include "ck/library/tensor_operation_instance/gpu/gemm.hpp"

using F16 = ck::half_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

using ADataType = F32;
using BDataType = F32;
using CDataType = F32;

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using DeviceOp =
    ck::tensor_operation::device::DeviceGemm<ALayout,
                                                BLayout,
                                                CLayout,
                                                ADataType,
                                                BDataType,
                                                CDataType,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough,
                                                ck::tensor_operation::element_wise::PassThrough>;

const auto op_ptrs = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
        DeviceOp>::GetInstances();         

const auto a_element_op = AElementOp{};
const auto b_element_op = BElementOp{};
const auto c_element_op = CElementOp{};

extern "C" __attribute__((visibility("default"))) int num_versions() {
    return op_ptrs.size();
}

extern "C" __attribute__((visibility("default"))) char* version_name(int ver) {
    if (ver >= 0 && ver < op_ptrs.size()) {
        std::string name = op_ptrs[ver]->GetTypeString();
        char *d = name.data();
        char *d1 = new char[name.length() + 1];
        strcpy(d1, d);
        return d1;
    }
    return nullptr;
}

extern "C" __attribute__((visibility("default"))) bool ck_gemm_f32(float* a, float* b, float* c, int m, int k, int n, int ver) {
    if (ver >= 0 && ver < op_ptrs.size()) {
        auto& op_ptr = op_ptrs[ver];
        auto argument_ptr = op_ptr->MakeArgumentPointer(a,
                                                              b,
                                                              c,
                                                              m,
                                                              n,
                                                              k,
                                                              k,
                                                              n,
                                                              n,
                                                              a_element_op,
                                                              b_element_op,
                                                              c_element_op);
        auto invoker_ptr = op_ptr->MakeInvokerPointer();                                                  

        invoker_ptr->Run(argument_ptr.get(), StreamConfig{nullptr, false});
        return true;
    }
    return false;
}