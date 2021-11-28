#pragma once
#include "Common/Tensor.h"

Tensor PerElementMult(Tensor& left, Tensor& right);
Tensor PerElementDiv(Tensor& left, Tensor& right);

Tensor MatrixMult(Tensor& left, uint channelLeft, Tensor& right, uint channelRight);
Tensor TransposeMatrix(Tensor& value, uint channel);


double Sum(Tensor& value);
Tensor SumForChannels(Tensor& value);
Tensor ReduceCols(Tensor& value);

Tensor ApplyReLU(Tensor& value);
Tensor ApplyExp(Tensor& value);
Tensor ApplySqrt(Tensor& value);