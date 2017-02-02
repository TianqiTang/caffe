#include <cfloat>

#include "caffe/common.hpp"
#include "caffe/util/fix_point.hpp"

namespace caffe {

template <typename Dtype>
void caffe_cpu_fix(const int n, const Dtype*x, Dtype* y, const int bit_level, const int scale) {
  Dtype step=std::pow(Dtype(2), scale-bit_level);
  Dtype lower_bound=-std::pow(Dtype(2), scale);
  Dtype upper_bound=std::pow(Dtype(2), scale)-step;
  for(int i=0; i<n; i++) {
    if(x[i] >= 0) {
       y[i]=std::min(std::max((Dtype)std::floor(x[i]/step+0.5)*step, lower_bound), upper_bound);
    }
    else {
      y[i]=std::min(std::max((Dtype)std::ceil(x[i]/step-0.5)*step, lower_bound), upper_bound);
    }
  }
}
template void caffe_cpu_fix<float>(const int n, const float* x, float* y, const int bit_level, const int scale);
template void caffe_cpu_fix<double>(const int n, const double* x, double* y, const int bit_level, const int scale);

}  // namespace caffe
