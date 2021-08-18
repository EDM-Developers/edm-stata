#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>

#ifndef FMT_HEADER_ONLY
#define FMT_HEADER_ONLY
#endif
#include <fmt/format.h>

#include "edm.h"
#include "manifold.h"

std::vector<bool> trueVec(size_t size)
{
  std::vector<bool> v(size);
  for (int i = 0; i < v.size(); i++) {
    v[i] = true;
  }
  return v;
}

void print_manifold(const Manifold& M)
{
  auto stringVersion = [](double v) { return (v == MISSING) ? std::string(" . ") : fmt::format("{:.1f}", v); };

  std::cout << "\n";
  for (int i = 0; i < M.nobs(); i++) {
    for (int j = 0; j < M.E_actual(); j++) {
      std::cout << stringVersion(M(i, j)) << "\t";
    }
    std::cout << "\t|\t" << stringVersion(M.y(i)) << "\n";
  }
  std::cout << "\n";
}

void require_manifolds_match(const Manifold& M, const std::vector<std::vector<double>>& M_true,
                             const std::vector<double>& y_true)
{
  for (int i = 0; i < M.nobs(); i++) {
    CAPTURE(i);
    for (int j = 0; j < M.E_actual(); j++) {
      CAPTURE(j);
      REQUIRE(M(i, j) == M_true[i][j]);
    }
    REQUIRE(M.y(i) == y_true[i]);
  }
}

TEST_CASE("Basic manifold creation", "[basicManifold]")
{
  int E = 2;
  int tau = 1;
  int p = 1;

  std::vector<double> t = { 1, 2, 3, 4 };
  std::vector<double> x = { 11, 12, 13, 14 };

  SECTION("Basic manifold, no extras or dt")
  {
    ManifoldGenerator generator(t, x, tau, p);

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E);
    REQUIRE(usable.size() == 4); // Not allowing first point because x.at(-1) doesn't exist
    REQUIRE(usable[0] == false);
    REQUIRE(usable[1] == true);
    REQUIRE(usable[2] == true);
    REQUIRE(usable[3] == false); // TODO: Not letting the last point be usable because y(end) is missing.. change?

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 2);  // TODO: Rename this to numPoints
    REQUIRE(M.ySize() == 2); // Shouldn't this always be the same as M.nobs()?
    REQUIRE(M.E_actual() == 2);

    std::vector<std::vector<double>> M_true = { { 12.0, 11.0 }, { 13.0, 12.0 } };
    std::vector<double> y_true = { 13.0, 14.0 };
    require_manifolds_match(M, M_true, y_true);
  }

  SECTION("Manifold with dt (not allowing missing)")
  {
    // TODO: This test is a bit fake; edm.ado would not allow dt to be
    // applied when there's no gaps in the time variable.
    bool dtWeight = 1.0;
    bool allowMissing = false;
    ManifoldGenerator generator(t, x, tau, p, {}, {}, {}, {}, 0, dtWeight, true, false, allowMissing);

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E);

    REQUIRE(usable.size() == 4);
    REQUIRE(usable[0] == false); // Not allowing first point because x.at(-1) doesn't exist
    REQUIRE(usable[1] == true);
    REQUIRE(usable[2] == true);
    REQUIRE(usable[3] ==
            false); // Not letting the last point be usable because y(end) is missing & because dt0 is missing

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 2);
    REQUIRE(M.ySize() == 2);
    REQUIRE(M.E_actual() == 4);

    std::vector<std::vector<double>> M_true = { { 12.0, 11.0, 1.0, 1.0 }, { 13.0, 12.0, 1.0, 1.0 } };
    std::vector<double> y_true = { 13.0, 14.0 };
    require_manifolds_match(M, M_true, y_true);
  }
}

// These tests are used as examples in the Julia docs
TEST_CASE("Missing data manifold creation (tau = 1)", "[missingDataManifold]")
{
  int E = 2;
  int tau = 1;
  int p = 1;

  std::vector<double> t = { 1.0, 2.5, 3.0, 4.5, 5.0, 6.0 };
  std::vector<double> x = { 11, 12, MISSING, 14, 15, 16 };

  SECTION("Default")
  {
    ManifoldGenerator generator(t, x, tau, p);

    REQUIRE(generator.calculate_time_increment() == 0.5);

    std::vector<int> obsNums = { 0, 3, 4, 7, 8, 10 };
    for (int i = 0; i < obsNums.size(); i++) {
      CAPTURE(i);
      REQUIRE(generator.get_observation_num(i) == obsNums[i]);
    }

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E);
    REQUIRE(usable.size() == 6);
    REQUIRE(usable[0] == false); // x is missing
    REQUIRE(usable[1] == false); // y is missing
    REQUIRE(usable[2] == false); // x is missing
    REQUIRE(usable[3] == false); // x is missing
    REQUIRE(usable[4] == false); // y is missing
    REQUIRE(usable[5] == false); // y is missing

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 0);
    REQUIRE(M.ySize() == 0);
    REQUIRE(M.E_actual() == 2);
  }

  SECTION("dt")
  {
    bool dtWeight = 1.0;
    bool allowMissing = false;
    ManifoldGenerator generator(t, x, tau, p, {}, {}, {}, {}, 0, dtWeight, true, false, allowMissing);

    std::vector<int> obsNums = { 0, 1, -1, 2, 3, 4 };
    for (int i = 0; i < obsNums.size(); i++) {
      CAPTURE(i);
      REQUIRE(generator.get_observation_num(i) == obsNums[i]);
    }

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E);
    REQUIRE(usable.size() == 6);
    REQUIRE(usable[0] == false); // x.at(-1) missing and nothing before it to bring forward
    REQUIRE(usable[1] == true);
    REQUIRE(usable[2] == false); // x is missing (can be skipped over in next point)
    REQUIRE(usable[3] == true);
    REQUIRE(usable[4] == true);
    REQUIRE(usable[5] == false); // dt0 and y is missing

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 3);
    REQUIRE(M.ySize() == 3);
    REQUIRE(M.E_actual() == 4);

    std::vector<std::vector<double>> M_true = { { 12.0, 11.0, 2.0, 1.5 },
                                                { 14.0, 12.0, 0.5, 2.0 },
                                                { 15.0, 14.0, 1.0, 0.5 } };
    std::vector<double> y_true = { 14.0, 15.0, 16.0 };
    require_manifolds_match(M, M_true, y_true);
  }

  SECTION("dt and allowingmissing")
  {
    bool dtWeight = 1.0;
    bool allowMissing = true;
    ManifoldGenerator generator(t, x, tau, p, {}, {}, {}, {}, 0, dtWeight, true, false, allowMissing);

    std::vector<int> obsNums = { 0, 1, 2, 3, 4, 5 };
    for (int i = 0; i < obsNums.size(); i++) {
      CAPTURE(i);
      REQUIRE(generator.get_observation_num(i) == obsNums[i]);
    }

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E);
    REQUIRE(usable.size() == 6);
    REQUIRE(usable[0] == true);
    REQUIRE(usable[1] == false); // y is missing
    REQUIRE(usable[2] == true);
    REQUIRE(usable[3] == true);
    REQUIRE(usable[4] == true);
    REQUIRE(usable[5] == false); // y is missing

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 4);
    REQUIRE(M.ySize() == 4);
    REQUIRE(M.E_actual() == 4);

    std::vector<std::vector<double>> M_true = { { 11.0, MISSING, 1.5, MISSING },
                                                { MISSING, 12.0, 1.5, 0.5 },
                                                { 14.0, MISSING, 0.5, 1.5 },
                                                { 15.0, 14.0, 1.0, 0.5 } };
    std::vector<double> y_true = { 12.0, 14.0, 15.0, 16.0 };
    require_manifolds_match(M, M_true, y_true);
  }
}

TEST_CASE("Missing data dt manifold creation (tau = 2)", "[missingDataManifold2]")
{
  int E = 2;
  int tau = 2;
  int p = 1;

  std::vector<double> t = { 1.0, 2.5, 3.0, 4.5, 5.0, 6.0 };
  std::vector<double> x = { 11, 12, MISSING, 14, 15, 16 };

  SECTION("Allowing missing values")
  {
    bool dtWeight = 1.0;
    bool allowMissing = true;
    ManifoldGenerator generator(t, x, tau, p, {}, {}, {}, {}, 0, dtWeight, true, false, allowMissing);

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E);
    REQUIRE(usable.size() == 6);
    REQUIRE(usable[0] == true);
    REQUIRE(usable[1] == false); // y is missing
    REQUIRE(usable[2] == true);
    REQUIRE(usable[3] == true);
    REQUIRE(usable[4] == true);
    REQUIRE(usable[5] == false); // y is missing

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 4);
    REQUIRE(M.ySize() == 4);
    REQUIRE(M.E_actual() == 4);

    std::vector<std::vector<double>> M_true = {
      { 11.0, MISSING, 1.5, MISSING },
      //  { 12.0, MISSING, 0.5, MISSING},
      { MISSING, 11.0, 1.5, 2.0 },
      { 14.0, 12.0, 0.5, 2.0 },
      { 15.0, MISSING, 1.0, 2.0 },
      //  { 16.0, 14.0, MISSING, 1.5},
    };
    std::vector<double> y_true = { 12.0, 14.0, 15.0, 16.0 };
    require_manifolds_match(M, M_true, y_true);
  }

  SECTION("Not allowing missing values")
  {
    bool dtWeight = 1.0;
    bool allowMissing = false;
    ManifoldGenerator generator(t, x, tau, p, {}, {}, {}, {}, 0, dtWeight, true, false, allowMissing);

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E);
    REQUIRE(usable.size() == 6);
    REQUIRE(usable[0] == false); // x.at(-1) missing and nothing before it to bring forward
    REQUIRE(usable[1] == false); // y is missing
    REQUIRE(usable[2] == false); // can't start on missing x
    REQUIRE(usable[3] == true);
    REQUIRE(usable[4] == true);
    REQUIRE(usable[5] == false); // dt0 and y is missing

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 2);
    REQUIRE(M.ySize() == 2);
    REQUIRE(M.E_actual() == 4);

    std::vector<std::vector<double>> M_true = { { 14.0, 11.0, 0.5, 3.5 }, { 15.0, 12.0, 1.0, 2.5 } };
    std::vector<double> y_true = { 15.0, 16.0 };
    require_manifolds_match(M, M_true, y_true);
  }
}

TEST_CASE("Check negative times work", "[negativeTimes]")
{
  std::vector<double> t = { -9.0, -7.5, -7.0, -6.5, -5.0, -4.0 };
  std::vector<double> x = { 11, 12, MISSING, 14, 15, 16 };

  int tau = 1;
  int p = 1;
  int E = 2;

  std::vector<std::vector<double>> extras;
  int numExtrasLagged = 0;

  ManifoldGenerator generator(t, x, tau, p);

  REQUIRE(generator.calculate_time_increment() == 0.5);

  std::vector<int> obsNums = { 0, 3, 4, 5, 8, 10 };
  for (int i = 0; i < obsNums.size(); i++) {
    CAPTURE(i);
    REQUIRE(generator.get_observation_num(i) == obsNums[i]);
  }
}