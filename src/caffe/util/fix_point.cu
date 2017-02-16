//#include <mutex>
#include <cfloat>

#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/fix_point.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void saturate_fix_kernel(const int n, Dtype upper_bound, Dtype lower_bound, Dtype step, const Dtype* x, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
	y[index]=min(max(round(x[index]/step)*step, lower_bound), upper_bound);
  }
}

template <typename Dtype>
void caffe_gpu_fix(const int n, const Dtype*x, Dtype* y, const int bit_level, const int scale) {
	Dtype step=std::pow(Dtype(2), scale-bit_level);
	Dtype lower_bound=-std::pow(Dtype(2), scale);
	Dtype upper_bound=std::pow(Dtype(2), scale)-step;
	saturate_fix_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
			n, upper_bound, lower_bound, step, x, y);
	CUDA_POST_KERNEL_CHECK;
}
template void caffe_gpu_fix<float>(const int n, const float* x, float* y, const int bit_level, const int scale);
template void caffe_gpu_fix<double>(const int n, const double* x, double* y, const int bit_level, const int scale);

/*
template <typename Dtype>
void caffe_gpu_fix_diff_w(const int n, const Dtype*x, Dtype* y, const int bit_level, const int max_scale, const int min_scale, int& final_scale) {
	final_scale=max_scale;
	Dtype fixed_diff_min = FLT_MAX;
	for(int scale=max_scale; scale>=min_scale; scale--){
		caffe_gpu_fix<Dtype>(n, x, y, bit_level, scale);
		caffe_gpu_sub<Dtype>(n, x, y, y);
		Dtype fixed_diff;
		caffe_gpu_asum(n, y, &fixed_diff);
		if(fixed_diff<fixed_diff_min){
			final_scale=scale;
			fixed_diff_min=fixed_diff;
		}
	}
	caffe_gpu_fix(n, x, y, bit_level, final_scale);
}
template void caffe_gpu_fix_diff_w<float>(const int n, const float* x, float* y, const int bit_level, const int max_scale, const int min_scale, int& final_scale);
template void caffe_gpu_fix_diff_w<double>(const int n, const double* x, double* y, const int bit_level, const int max_scale, const int min_scale, int& final_scale);

template <typename Dtype>
static __global__ void overflow_kernel(const int n, Dtype upper_bound, Dtype lower_bound, const Dtype* x, Dtype* y) {
	CUDA_KERNEL_LOOP(index, n) {
		y[index]=(x[index]<=upper_bound && x[index]>=lower_bound)?Dtype(0):Dtype(1);
	}
}

template <typename Dtype>
static bool test_overflow(const int n, Dtype upper_bound, Dtype lower_bound, const Dtype* data, Dtype* buffer) {
	overflow_kernel<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(n, upper_bound, lower_bound, data, buffer);
	CUDA_POST_KERNEL_CHECK;
	Dtype asum;
	caffe_gpu_asum(n, buffer, &asum);
	return asum>Dtype(0.5);
}

template <typename Dtype>
void caffe_gpu_fix_overflow(const int n, const Dtype* x, Dtype* y, const int bit_level, const int max_scale, const int min_scale, int& final_scale) {
	final_scale=std::max(std::min(final_scale, max_scale), min_scale);
	int search_length=max_scale-min_scale+1;
	if(search_length<2) {
		final_scale=min_scale;
	}
	else {
		Dtype* buffer=y;
		if(x==y) {
			buffer=static_cast<Dtype*>(Caffe::GpuBuffer(n*sizeof(Dtype)));
			CHECK_NOTNULL(buffer);
		}

		vector<Dtype> upper_bound(search_length);
		vector<Dtype> lower_bound(search_length);
		for(int i=0; i<search_length; i++) {
			upper_bound[i]=std::pow(Dtype(2), i+min_scale);
			lower_bound[i]=-upper_bound[i]-std::pow(Dtype(2), i+min_scale-bit_level);
		}

		vector<bool> overflow(search_length);
		vector<bool> tested(search_length, false);

		bool found=false;
		overflow[final_scale-min_scale]=test_overflow(n, upper_bound[final_scale-min_scale],
				lower_bound[final_scale-min_scale], x, buffer);
		tested[final_scale-min_scale]=true;
		if(!overflow[final_scale-min_scale]) {
			if(final_scale==min_scale) {
				found=true;
			}
			else {
				overflow[final_scale-min_scale-1]=test_overflow(n, upper_bound[final_scale-min_scale-1],
						lower_bound[final_scale-min_scale-1], x, buffer);
				tested[final_scale-min_scale-1]=true;
				if(overflow[final_scale-min_scale-1]) {
					found=true;
				}
			}
		}

		if(!found) {
			overflow[0]=true;
			tested[0]=true;
			overflow[search_length-1]=false;
			tested[search_length-1]=true;
			int left=0;
			int right=search_length-1;
			for(;;) {
				int middle=(left+right)/2;
				if(!tested[middle]) {
					overflow[middle]=test_overflow(n, upper_bound[middle], lower_bound[middle], x, buffer);
					tested[middle]=true;
				}
				if(!tested[middle+1]) {
					overflow[middle+1]=test_overflow(n, upper_bound[middle+1], lower_bound[middle+1], x, buffer);
					tested[middle+1]=true;
				}
				if(overflow[middle] && !overflow[middle+1]) {
					final_scale=min_scale+middle+1;
					break;
				}
				else if(!overflow[middle]) {
					right=middle;
				}
				else {
					left=middle+1;
				}
			}
		}
	}
	caffe_gpu_fix(n, x, y, bit_level, final_scale);
}
template void caffe_gpu_fix_overflow<float>(const int n, const float* x, float* y, const int bit_level, const int max_scale, const int min_scale, int& final_scale);
template void caffe_gpu_fix_overflow<double>(const int n, const double* x, double* y, const int bit_level, const int max_scale, const int min_scale, int& final_scale);
*/
}  // namespace caffe
