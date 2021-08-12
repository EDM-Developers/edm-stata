#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file
#include <catch2/catch.hpp>

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

TEST_CASE("Basic manifold creation", "[basicManifold]")
{
  std::vector<double> t = { 1, 2, 3, 4 };
  std::vector<double> x = { 11, 12, 13, 14 };
  std::vector<double> y = { 12, 13, 14, MISSING };

  std::vector<std::vector<double>> extras;
  int numExtrasLagged = 0;
  int tau = 1;

  ManifoldGenerator generator(t, x, y, extras, numExtrasLagged, MISSING, tau);

  SECTION("Basic manifold, no extras or dt")
  {

    int E = 2;

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E, false);
    REQUIRE(usable.size() == 4); // Not allowing first point because x.at(-1) doesn't exist
    REQUIRE(usable[0] == false);
    REQUIRE(usable[1] == true);
    REQUIRE(usable[2] == true);
    REQUIRE(usable[3] == false); // TODO: Not letting the last point be usable because y(end) is missing.. change?

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 2);  // TODO: Rename this to numPoints
    REQUIRE(M.ySize() == 2); // Shouldn't this always be the same as M.nobs()?
    REQUIRE(M.E_actual() == 2);

    REQUIRE(M(0, 0) == 12);
    REQUIRE(M(0, 1) == 11);

    REQUIRE(M(1, 0) == 13);
    REQUIRE(M(1, 1) == 12);
  }

  SECTION("Manifold with dt (not allowing missing)")
  {

    generator.add_dt_data(1.0, true, false);

    int E = 2;

    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E, false);

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

    REQUIRE(M(0, 0) == 12);
    REQUIRE(M(0, 1) == 11);
    REQUIRE(M(0, 2) == 1); // dt0
    REQUIRE(M(0, 3) == 1); // dt1

    REQUIRE(M(1, 0) == 13);
    REQUIRE(M(1, 1) == 12);
    REQUIRE(M(1, 2) == 1); // dt0
    REQUIRE(M(1, 3) == 1); // dt1
  }
}

TEST_CASE("Missing data dt manifold creation (tau = 1)", "[missingDataManifold]")
{
  std::vector<double> t = { 1.0, 2.5, 3.0, 4.5, 5.0, 6.0 };
  std::vector<double> x = { 11, 12, MISSING, 14, 15, 16 };
  int tau = 1;
  std::vector<double> y = { 12, MISSING, 14, 15, 16, MISSING };

  std::vector<std::vector<double>> extras;
  int numExtrasLagged = 0;

  std::vector<bool> allTrue(t.size());
  for (int i = 0; i < allTrue.size(); i++) {
    allTrue[i] = true;
  }

  int E = 2;

  ManifoldGenerator generator(t, x, y, extras, numExtrasLagged, MISSING, tau);
  generator.add_dt_data(1.0, true, false);

  SECTION("Allowing missing values")
  {
    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E, true);
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

    for (int i = 0; i < M.nobs(); i++) {
      for (int j = 0; j < M.E_actual(); j++) {
        REQUIRE(M(i, j) == M_true[i][j]);
      }
      REQUIRE(M.y(i) == y_true[i]);
    }
  }

  SECTION("Not allowing missing values")
  {
    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E, false);
    REQUIRE(usable.size() == 6);
    REQUIRE(usable[0] == false); // x.at(-1) missing and nothing before it to bring forward
    REQUIRE(usable[1] == false); // y is missing
    REQUIRE(usable[2] == false); // x is missing (can be skipped over in next point)
    REQUIRE(usable[3] == true);  //  <---- currently fails; needs to be able to skip over the previous missing value
    REQUIRE(usable[4] == true);
    REQUIRE(usable[5] == false); // dt0 and y is missing

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 2);
    REQUIRE(M.ySize() == 2);
    REQUIRE(M.E_actual() == 4);

    std::vector<std::vector<double>> M_true = { { 14.0, 12.0, 0.5, 2.0 }, { 15.0, 14.0, 1.0, 0.5 } };

    std::vector<double> y_true = { 15.0, 16.0 };

    for (int i = 0; i < M.nobs(); i++) {
      for (int j = 0; j < M.E_actual(); j++) {
        REQUIRE(M(i, j) == M_true[i][j]);
      }
      REQUIRE(M.y(i) == y_true[i]);
    }
  }
}

TEST_CASE("Missing data dt manifold creation (tau = 2)", "[missingDataManifold2]")
{
  std::vector<double> t = { 1.0, 2.5, 3.0, 4.5, 5.0, 6.0 };
  std::vector<double> x = { 11, 12, MISSING, 14, 15, 16 };
  int tau = 2;
  std::vector<double> y = { 12, MISSING, 14, 15, 16, MISSING };

  std::vector<std::vector<double>> extras;
  int numExtrasLagged = 0;

  int E = 2;

  ManifoldGenerator generator(t, x, y, extras, numExtrasLagged, MISSING, tau);
  generator.add_dt_data(1.0, true, false);

  SECTION("Allowing missing values")
  {
    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E, true);
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

    for (int i = 0; i < M.nobs(); i++) {
      for (int j = 0; j < M.E_actual(); j++) {
        REQUIRE(M(i, j) == M_true[i][j]);
      }
      REQUIRE(M.y(i) == y_true[i]);
    }
  }

  SECTION("Not allowing missing values")
  {
    std::vector<bool> usable = generator.generate_usable(trueVec(t.size()), E, false);
    REQUIRE(usable.size() == 6);
    REQUIRE(usable[0] == false); // x.at(-1) missing and nothing before it to bring forward
    REQUIRE(usable[1] == false); // y is missing
    REQUIRE(usable[2] == false); // can't start on missing x
    REQUIRE(usable[3] == true);  //  <---- currently fails; needs to be able to skip over the previous missing value
    REQUIRE(usable[4] == true);
    REQUIRE(usable[5] == false); // dt0 and y is missing

    Manifold M = generator.create_manifold(E, usable, false, false, false);

    REQUIRE(M.nobs() == 2);
    REQUIRE(M.ySize() == 2);
    REQUIRE(M.E_actual() == 4);

    std::vector<std::vector<double>> M_true = { { 14.0, 12.0, 0.5, 2.0 }, { 15.0, 14.0, 1.0, 0.5 } };

    std::vector<double> y_true = { 15.0, 16.0 };

    for (int i = 0; i < M.nobs(); i++) {
      for (int j = 0; j < M.E_actual(); j++) {
        REQUIRE(M(i, j) == M_true[i][j]);
      }
      REQUIRE(M.y(i) == y_true[i]);
    }
  }
}
