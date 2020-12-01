#pragma once

#include <vector>

class Manifold
{
private:
  bool _use_dt;
  double _dtWeight, _missing;
  size_t _nobs, _E_x, _E_dt, _E_extras, _E_actual;

  std::vector<double> _x, _t;
  std::vector<std::vector<double>> _extras;

  std::vector<bool> _filter;
  std::vector<size_t> _filteredIndices;

  mutable std::vector<double> _x_flat;
  mutable std::vector<double> _dt_flat;
  mutable std::vector<double> _extras_flat;
  mutable std::vector<double> _combined_flat;

  double find_x(size_t i, size_t j) const;
  double find_dt(size_t i, size_t j) const;
  double find_extras(size_t i, size_t j) const;

public:
  Manifold(std::vector<double> x, std::vector<double> t, std::vector<std::vector<double>> extras,
           std::vector<bool> filter, size_t E, double dtWeight, double missing);

  void set_filter(std::vector<bool> filter);

  void compute_lagged_embedding() const;

  double operator()(size_t i, size_t j) const { return _combined_flat[i * _E_actual + j]; }

  void compute_lagged_embeddings() const;

  double x(size_t i, size_t j) const { return _x_flat[i * _E_x + j]; }

  double dt(size_t i, size_t j) const { return _dt_flat[i * _E_dt + j]; }

  double extras(size_t i, size_t j) const { return _extras_flat[i * _E_extras + j]; }

  bool any_missing(size_t obsNum) const;

  std::vector<bool> get_filter() const { return _filter; };
  size_t nobs() const { return _nobs; }
  size_t E() const { return _E_x; }
  size_t E_dt() const { return _E_dt; }
  size_t E_extra() const { return _E_extras; }
  size_t E_actual() const { return _E_actual; }
};