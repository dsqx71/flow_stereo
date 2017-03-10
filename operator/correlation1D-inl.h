/*!
 * Copyright (c) 2015 by Contributors
 * \file correlation1D-inl.h
 * \brief correlation1D operator and symbol
 * \author Xu Dong
*/
#ifndef MXNET_OPERATOR_CORRELATION1D_INL_H_
#define MXNET_OPERATOR_CORRELATION1D_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./mshadow_op.h"
#include "./operator_common.h"
namespace mxnet {
namespace op {
//  Declare enumeration of input order to make code more intuitive.
//  These enums are only visible within this header
namespace Correlation1D {
enum  Correlation1DOpInputs{kData1, kData2};
enum  Correlation1DOpOutputs{kOut, kTemp1, kTemp2};
}  //  namespace Correlation1D
struct Correlation1DParam : public dmlc::Parameter<Correlation1DParam> {
  uint32_t max_displacement;
  uint32_t kernel_size;
  uint32_t pad_size;
  uint32_t stride1;
  uint32_t stride2;
  uint32_t single_side;
  DMLC_DECLARE_PARAMETER(Correlation1DParam) {
    DMLC_DECLARE_FIELD(kernel_size).set_default(1)
    .describe("kernel size for Correlation1D must be an odd number");
    DMLC_DECLARE_FIELD(max_displacement).set_default(1)
    .describe("Max displacement of Correlation1D ");
    DMLC_DECLARE_FIELD(stride1).set_default(1)
    .describe("stride1 quantize data1 globally");
    DMLC_DECLARE_FIELD(stride2).set_default(1)
    .describe("stride2 quantize data2 within the neighborhood centered around data1");
    DMLC_DECLARE_FIELD(pad_size).set_default(0)
    .describe("pad for Correlation1D");
    DMLC_DECLARE_FIELD(single_side).set_default(0)
    .describe("0: both side, -1: to left, 1: to right");
  }
};
template<typename xpu>
class Correlation1DOp : public Operator {
 public:
  explicit Correlation1DOp(Correlation1DParam param) {
    this->param_ = param;
  }
  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> data1 = in_data[Correlation1D::kData1].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> data2 = in_data[Correlation1D::kData2].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out   = out_data[Correlation1D::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tmp1  = out_data[Correlation1D::kTemp1].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tmp2  = out_data[Correlation1D::kTemp2].get<xpu, 4, real_t>(s);
    tmp1 = 0.0f;
    tmp2 = 0.0f;
    out = 0.0f;
    CHECK_EQ(data1.CheckContiguous(), true);
    CHECK_EQ(data2.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    CHECK_EQ(tmp1.CheckContiguous(), true);
    CHECK_EQ(tmp2.CheckContiguous(), true);

    paddedbottomheight = data1.shape_[2];
    paddedbottomwidth  = data1.shape_[3] + 2 * param_.pad_size;

    kernel_radius_ = (param_.kernel_size - 1) / 2;
    border_size_ = param_.max_displacement + kernel_radius_;
    stride1 = param_.stride1;
    stride2 = param_.stride2;
    top_width_ = ceil(static_cast<float>(paddedbottomwidth - border_size_ * 2)\
     / static_cast<float>(stride1));
    top_height_ = ceil(static_cast<float>(paddedbottomheight - kernel_radius_ * 2)\
     / static_cast<float>(stride1));
    neighborhood_grid_radius_ = param_.max_displacement / stride2;
    if (param_.single_side!=0)
       neighborhood_grid_width_ = neighborhood_grid_radius_ + 1;
    else
       neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;

    top_channels_ = neighborhood_grid_width_;
    num =  data1.shape_[0];
    channels = data1.shape_[1];
    height = data1.shape_[2];
    width = data1.shape_[3];
    Correlation1DForward(out, data1, data2, tmp1, tmp2, top_channels_, top_height_, top_width_,
                       param_.pad_size, param_.single_side,
                       param_.max_displacement, param_.kernel_size,
                       neighborhood_grid_radius_, neighborhood_grid_width_,
                       kernel_radius_, param_.stride1, param_.stride2);
  }
  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 4> grad_data1 = in_grad[Correlation1D::kData1].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> grad_data2 = in_grad[Correlation1D::kData2].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> out_g = out_grad[Correlation1D::kOut].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tmp1 = out_data[Correlation1D::kTemp1].get<xpu, 4, real_t>(s);
    Tensor<xpu, 4> tmp2 = out_data[Correlation1D::kTemp2].get<xpu, 4, real_t>(s);
    CHECK_EQ(grad_data1.CheckContiguous(), true);
    CHECK_EQ(grad_data2.CheckContiguous(), true);
    CHECK_EQ(out_g.CheckContiguous(), true);
    CHECK_EQ(tmp1.CheckContiguous(), true);
    CHECK_EQ(tmp2.CheckContiguous(), true);
    Correlation1DBackward(out_g, grad_data1, grad_data2, tmp1, tmp2, top_channels_,
    top_height_, top_width_, param_.pad_size, param_.single_side,
    param_.max_displacement, param_.kernel_size, neighborhood_grid_radius_,
    neighborhood_grid_width_, kernel_radius_, param_.stride1, param_.stride2,
    num, channels, height, width);
  }

 private:
    Correlation1DParam param_;
    int paddedbottomheight;
    int paddedbottomwidth;
    uint32_t kernel_radius_;
    uint32_t border_size_;
    uint32_t stride1;
    uint32_t stride2;
    uint32_t top_width_;
    uint32_t top_height_;
    uint32_t neighborhood_grid_radius_;
    uint32_t neighborhood_grid_width_;
    uint32_t top_channels_;
    int  num;
    int  channels;
    int  height;
    int  width;
};   //  class Correlation1DOp
//  Decalre Factory function
template<typename xpu>
Operator* CreateOp(Correlation1DParam param);
#if DMLC_USE_CXX11
class Correlation1DProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data1", "data2"};
  }
  std::vector<std::string> ListOutputs() const override {
    return {"output", "tmp1", "tmp2"};
  }
  int NumOutputs() const override {
    return 3;
  }
  int NumVisibleOutputs() const override {
    return 1;
  }
void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }
  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }
  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2) << "Input:[data1, data2]";
    TShape dshape1 = in_shape->at(Correlation1D::kData1);
    TShape dshape2 = in_shape->at(Correlation1D::kData2);
    CHECK_EQ(dshape1.ndim(), 4) << "data should be a 4D tensor";
    CHECK_EQ(dshape2.ndim(), 4) << "data should be a 4D tensor";
    int paddedbottomheight;
    int paddedbottomwidth;
    uint32_t kernel_radius_;
    uint32_t stride1;
    uint32_t stride2;
    uint32_t top_width_;
    uint32_t top_height_;
    uint32_t neighborhood_grid_radius_;
    uint32_t neighborhood_grid_width_;
    uint32_t top_channels_;
    uint32_t border_size_;
    paddedbottomheight = dshape1[2];
    paddedbottomwidth  = dshape1[3] + 2*param_.pad_size;
    kernel_radius_ = (param_.kernel_size -1)/2;
    border_size_ = param_.max_displacement + kernel_radius_;
    stride1 = param_.stride1;
    stride2 = param_.stride2;
    top_width_ = ceil(static_cast<float>(paddedbottomwidth - border_size_ * 2)\
     / static_cast<float>(stride1));
    top_height_ = ceil(static_cast<float>(paddedbottomheight - kernel_radius_ * 2)\
     / static_cast<float>(stride1));
    neighborhood_grid_radius_ = param_.max_displacement / stride2;
    if (param_.single_side!=0)
       neighborhood_grid_width_ = neighborhood_grid_radius_ + 1;
    else
       neighborhood_grid_width_ = neighborhood_grid_radius_ * 2 + 1;
    top_channels_ = neighborhood_grid_width_;
    CHECK_GE(top_width_, 1) <<
    "Correlation1D cannot be done with current settings.Neighborhood and kernel don't fit in blob";
    CHECK_GE(top_height_, 1) <<
    "Correlation1D cannot be done with current settings.Neighborhood and kernel don't fit in blob";
    out_shape->clear();
    out_shape->push_back(Shape4(dshape1[0], top_channels_, top_height_, top_width_));
    out_shape->push_back(Shape4(dshape1[0], paddedbottomheight, paddedbottomwidth, dshape1[1]));
    out_shape->push_back(Shape4(dshape1[0], paddedbottomheight, paddedbottomwidth, dshape1[1]));
    return true;
  }
  OperatorProperty* Copy() const override {
    Correlation1DProp* Correlation1D_sym = new Correlation1DProp();
    Correlation1D_sym->param_ = this->param_;
    return Correlation1D_sym;
  }
  std::string TypeString() const override {
    return "Correlation1D";
  }
  //  decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
     return {out_grad[Correlation1D::kOut],
     out_data[Correlation1D::kTemp1], out_data[Correlation1D::kTemp2]};
}
  Operator* CreateOperator(Context ctx) const override;

 private:
  Correlation1DParam param_;
};  //  class Correlation1DProp
#endif
}  //  namespace op
}  //  namespace mxnet
#endif  //  MXNET_OPERATOR_CORRELATION1D_INL_H_
