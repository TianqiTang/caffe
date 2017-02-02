#ifndef CAFFE_UTIL_TRUNC_HPP_
#define CAFFE_UTIL_TRUNC_HPP_

namespace caffe {

template <typename Dtype>
void caffe_cpu_fix(const int n, const Dtype*x, Dtype* y, const int bit_level, const int scale);

#ifndef CPU_ONLY

template <typename Dtype>
void caffe_gpu_fix(const int n, const Dtype*x, Dtype* y, const int bit_level, const int scale);

/*
template <typename Dtype>
void caffe_gpu_fix_diff_w(const int n, const Dtype* x, Dtype* y, const int bit_level, const int max_scale, const int min_scale, int &final_scale);

template <typename Dtype>
void caffe_gpu_fix_overflow(const int n, const Dtype* x, Dtype* y, const int bit_level, const int max_scale, const int min_scale, int& final_scale);
*/
#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_TRUNC_HPP_
