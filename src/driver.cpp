#include "driver.h"

void write_results(std::string fname_out, const EdmResult& res, bool save_mode, int varssv)
{
  hid_t fid = H5Fcreate(fname_out.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  H5LTset_attribute_int(fid, "/", "rc", &(res.rc), 1);

  hsize_t ystarLen[] = { (hsize_t)res.ystar.size() };
  H5LTmake_dataset_double(fid, "ystar", 1, ystarLen, res.ystar.data());

  if (save_mode) {
    hsize_t Bi_mapLen[] = { (hsize_t)(res.flat_Bi_map.size() / varssv), (hsize_t)varssv };
    H5LTmake_dataset_double(fid, "flat_Bi_map", 2, Bi_mapLen, res.flat_Bi_map.data());
  }

  H5Fclose(fid);
}

int main(int argc, char* argv[])
{

  if (argc != 2) {
    fprintf(stderr, "Usage: ./driver <fname>\n");
    return -1;
  }

  std::string fname_in(argv[1]);

  EdmInputs vars = read_dumpfile(fname_in);
  IO io = { display, error, flush };
  EdmResult res = mf_smap_loop(vars.opts, vars.y, vars.M, vars.Mp, vars.nthreads, io);

  std::size_t ext = fname_in.find_last_of(".");
  fname_in = fname_in.substr(0, ext);
  std::string fname_out = fname_in + "-out.h5";

  write_results(fname_out, res, vars.opts.save_mode, vars.opts.varssv);

  return res.rc;
}