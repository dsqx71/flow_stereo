/*!
 * Copyright (c) 2015 by Contributors
 * \file correlation1D.cc
 * \brief correlation1D op
 * \author Xu Dong
*/
#include "./correlation1D-inl.h"
#include "./mshadow_op.h"

namespace mshadow {
template<typename Dtype>
inline void Correlation1DForward(const Tensor<cpu, 4, Dtype> &out,
                               const Tensor<cpu, 4, Dtype> &data1,
                               const Tensor<cpu, 4, Dtype> &data2,
                               const Tensor<cpu, 4, Dtype> &tmp1,
                               const Tensor<cpu, 4, Dtype> &tmp2,
                               int top_channels_, int top_height_, int top_width_,
                               int pad_size_, int single_side,
                               int max_displacement_, int kernel_size_,
                               int neighborhood_grid_radius_, int neighborhood_grid_width_,
                               int  kernel_radius_, int stride1_, int stride2_)
                               {
        printf("No implementation");
}
template<typename Dtype>
inline void Correlation1DBackward(const Tensor<cpu, 4, Dtype> &out_grad,
                                const Tensor<cpu, 4, Dtype> &in_grad1,
                                const Tensor<cpu, 4, Dtype> &in_grad2,
                                const Tensor<cpu, 4, Dtype> &tmp1,
                                const Tensor<cpu, 4, Dtype> &tmp2,
                                int top_channels_, int top_height_,
                                int top_width_, int pad_size_,
                                int single_side, int max_displacement_,
                                int kernel_size_, int neighborhood_grid_radius_,
                                int neighborhood_grid_width_,
                                int  kernel_radius_, int stride1_,
                                int stride2_, int num,
                                int channels, int height, int width
                            ) {
            printf("No implementation");
          }
}  // namespace mshadow
namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(Correlation1DParam param) {
  return new Correlation1DOp<cpu>(param);
}
Operator* Correlation1DProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}
DMLC_REGISTER_PARAMETER(Correlation1DParam);
MXNET_REGISTER_OP_PROPERTY(Correlation1D,Correlation1DProp)
.describe("Apply correlation1D to inputs")
.add_argument("data1", "Symbol", "Input data1 to the correlation1D.")
.add_argument("data2", "Symbol", "Input data2 to the correlation1D.")
.add_arguments(Correlation1DParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
