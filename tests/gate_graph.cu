// GATE: the CUDA-graph decode path must be BITWISE identical to the eager path.
//
// This is a prerequisite for the Phase 7 losslessness test, which compares
// speculation on/off. If the graph and eager paths already disagree, that test
// measures nothing.
#include <cstdio>
#include <vector>
#include <cstring>
#include <cuda_runtime.h>
#include "../src/model/model.h"
#include "../src/kernels/elementwise.cuh"

int main(int argc, char** argv) {
  const std::string md = argc > 1 ? argv[1]
      : "/home/patrickd/qwen38-weights/Qwen3.8-27B-W4A16-AWQ";
  qwen::Model m; qwen::LoadOptions o;
  o.max_ctx = 2048; o.max_batch = 64; o.verbose = false;
  qwen::model_load(m, md, o);
  qwen::model_graph_capture(m);

  const std::vector<int32_t> prompt = {785, 6722, 315, 9625, 374};
  const int N = 24;
  std::vector<uint16_t> lg_eager, lg_graph;
  std::vector<int32_t> tok_eager, tok_graph;

  for (int mode = 0; mode < 2; ++mode) {
    m.use_graph = (mode == 1);
    cudaMemset(m.gdn_state, 0, size_t(m.shape.gdn_state_elems()) * 4);
    cudaMemset(m.gdn_conv, 0, size_t(m.shape.gdn_conv_state_elems()) * 4);
    qwen::model_prefill(m, prompt.data(), int(prompt.size()), 0);
    int32_t* d_id = m.argmax_scratch + 512;
    auto& toks = mode ? tok_graph : tok_eager;
    auto& lgs  = mode ? lg_graph  : lg_eager;
    for (int i = 0; i < N; ++i) {
      qwen::argmax(d_id, m.logits, m.shape.vocab_size, m.argmax_scratch);
      int32_t t; cudaMemcpy(&t, d_id, 4, cudaMemcpyDeviceToHost);
      toks.push_back(t);
      if (i == N - 1) {
        lgs.resize(m.shape.vocab_size);
        cudaMemcpy(lgs.data(), m.logits, lgs.size() * 2, cudaMemcpyDeviceToHost);
      }
      qwen::model_decode(m, t, int(prompt.size()) + i);
    }
  }

  size_t tdiff = 0;
  for (int i = 0; i < N; ++i) tdiff += (tok_eager[i] != tok_graph[i]);
  size_t ldiff = 0;
  for (size_t i = 0; i < lg_eager.size(); ++i) ldiff += (lg_eager[i] != lg_graph[i]);

  printf("gate_graph\n");
  printf("  tokens differing : %zu / %d\n", tdiff, N);
  printf("  logit bits differing at the last step : %zu / %zu\n", ldiff, lg_eager.size());
  const bool pass = tdiff == 0 && ldiff == 0;
  printf("  RESULT : %s (bitwise)\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
