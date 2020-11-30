#pragma once

#include <unordered_map>
#include <vector>

class Manifold
{
private:
  bool _use_dt, _full_t;
  double _dtweight, _missing;
  size_t _nobs, _E_x, _E_dt, _E_extras, _E_actual;

  std::vector<double> _x;
  std::vector<int> _t;
  std::vector<std::vector<double>> _extras;

  std::vector<bool> _filter;
  std::vector<int> _filtered_t;

  std::vector<double> _x_flat;
  std::vector<double> _dt_flat;
  std::vector<double> _extras_flat;
  std::vector<double> _combined_flat;

  std::unordered_map<int, size_t> _timeToIndex;

  int obs_num_to_time(size_t obsNum) const;
  size_t time_to_index(int time) const;

  double find_x(size_t i, size_t j) const;
  double find_dt(size_t i, size_t j) const;
  double find_extras(size_t i, size_t j) const;

  void compute_lagged_embedding();
  void compute_lagged_embeddings();

public:
  Manifold(std::vector<double> x, std::vector<int> t, std::vector<std::vector<double>> extras, std::vector<bool> filter,
           size_t E, double dtweight, double missing);

  double x(size_t i, size_t j) const;
  double dt(size_t i, size_t j) const;
  double extras(size_t i, size_t j) const;
  double operator()(size_t i, size_t j) const;

  bool any_missing(size_t obsNum) const;

  void set_filter(std::vector<bool> filter);
  std::vector<bool> get_filter() const { return _filter; };
  size_t nobs() const { return _nobs; }
  size_t E() const { return _E_x; }
  size_t E_dt() const { return _E_dt; }
  size_t E_extra() const { return _E_extras; }
  size_t E_actual() const { return _E_actual; }
};