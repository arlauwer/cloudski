/*
  Since the hnsw lib in python can only handle floats, we export the hnsw
  points as floats to a numpy readable format.
*/

#include "hnsw/hnswlib.h"
#include <cmath>

namespace hnswlib {
class CloudySpace : public SpaceInterface<double> {
  DISTFUNC<double> fstdistfunc_;
  size_t data_size_;
  size_t dim_;

public:
  CloudySpace(size_t dim, DISTFUNC<double> dist_func) {
    fstdistfunc_ = dist_func;

    dim_ = dim;
    data_size_ = dim * sizeof(double);
  }

  size_t get_data_size() override { return data_size_; }

  DISTFUNC<double> get_dist_func() override { return fstdistfunc_; }

  void *get_dist_func_param() override { return &dim_; }

  ~CloudySpace() {}
};
}; // namespace hnswlib

constexpr int numBins = 24;
constexpr int dim = 2 + numBins;
const std::string location =
    "/home/arno/Documents/PhD/run/SKIRT/knn/AGN/lib/hnsw.bin";

constexpr double hden_factor = 1e0;
constexpr double metallicity_factor = 1e0;
constexpr double rad_factor = 1e0;

// cloudy distance function
static double cloudy_dist(const void *pVect1v, const void *pVect2v,
                          const void * /*qty_ptr*/) {
  double *pVect1 = (double *)pVect1v;
  double *pVect2 = (double *)pVect2v;
  // size_t qty = *((size_t*)qty_ptr);

  double res = 0;

  // log n
  // fix this: n1 nor n2 should ever be zero!!!!
  if (*pVect1 != 0. && *pVect2 != 0.)
    res += hden_factor * std::abs(std::log10(*pVect1) - std::log10(*pVect2));
  pVect1++;
  pVect2++;

  // linear Z
  if (*pVect1 != 0. && *pVect2 != 0.)
    res += metallicity_factor * std::abs(*pVect1 - *pVect2);
  pVect1++;
  pVect2++;

  // log rad
  for (size_t i = 0; i < numBins; i++) {
    // should never be 0!
    res += rad_factor * std::abs(std::log10(*pVect1) - std::log10(*pVect2));
    pVect1++;
    pVect2++;
  }
  return res;
}

int main() {

  auto space = new hnswlib::CloudySpace(dim, cloudy_dist);
  hnswlib::HierarchicalNSW<double> index(space, location);

  int els = index.getCurrentElementCount();
  auto t = index.size_data_per_element_;
  std::cout << t << std::endl;

  std::cout << els << std::endl;
  for (int i = 0; i < els; i++) {
    double *data = reinterpret_cast<double *>(index.getDataByInternalId(i));
    for (int j = 0; j < dim; j++) {
      std::cout << data[j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}