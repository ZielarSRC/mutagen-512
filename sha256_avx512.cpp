#include <immintrin.h>
#include <stdint.h>
#include <string.h>

#include "sha256_avx512.h"

namespace {

#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

ALIGN64 static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline __m512i rotr(const __m512i& x, int n) {
  return _mm512_or_si512(_mm512_srli_epi32(x, n), _mm512_slli_epi32(x, 32 - n));
}

inline __m512i shr(const __m512i& x, int n) { return _mm512_srli_epi32(x, n); }

inline void initialize(__m512i* state) {
  state[0] = _mm512_set1_epi32(0x6a09e667);
  state[1] = _mm512_set1_epi32(0xbb67ae85);
  state[2] = _mm512_set1_epi32(0x3c6ef372);
  state[3] = _mm512_set1_epi32(0xa54ff53a);
  state[4] = _mm512_set1_epi32(0x510e527f);
  state[5] = _mm512_set1_epi32(0x9b05688c);
  state[6] = _mm512_set1_epi32(0x1f83d9ab);
  state[7] = _mm512_set1_epi32(0x5be0cd19);
}

inline void prepare_message_schedule(const uint8_t* const data[16], __m512i* W) {
  ALIGN64 uint32_t words[16];
  for (int t = 0; t < 16; ++t) {
    for (int lane = 0; lane < 16; ++lane) {
      const uint8_t* ptr = data[lane] + t * 4;
      words[lane] = (static_cast<uint32_t>(ptr[0]) << 24) |
                    (static_cast<uint32_t>(ptr[1]) << 16) |
                    (static_cast<uint32_t>(ptr[2]) << 8) |
                    (static_cast<uint32_t>(ptr[3]));
    }
    W[t] = _mm512_load_si512(words);
  }
}

inline void transform(__m512i* state, const uint8_t* const data[16]) {
  __m512i a = state[0];
  __m512i b = state[1];
  __m512i c = state[2];
  __m512i d = state[3];
  __m512i e = state[4];
  __m512i f = state[5];
  __m512i g = state[6];
  __m512i h = state[7];

  ALIGN64 __m512i W[16];
  prepare_message_schedule(data, W);

#define S0(x)                                                                  \
  (_mm512_xor_si512(rotr(x, 2),                                                \
                    _mm512_xor_si512(rotr(x, 13), rotr(x, 22))))
#define S1(x)                                                                  \
  (_mm512_xor_si512(rotr(x, 6),                                                \
                    _mm512_xor_si512(rotr(x, 11), rotr(x, 25))))
#define s0(x)                                                                  \
  (_mm512_xor_si512(rotr(x, 7),                                                \
                    _mm512_xor_si512(rotr(x, 18), shr(x, 3))))
#define s1(x)                                                                  \
  (_mm512_xor_si512(rotr(x, 17),                                               \
                    _mm512_xor_si512(rotr(x, 19), shr(x, 10))))
#define Ch(x, y, z)                                                            \
  _mm512_xor_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z))
#define Maj(x, y, z)                                                           \
  _mm512_or_si512(_mm512_and_si512(x, y),                                      \
                  _mm512_and_si512(z, _mm512_or_si512(x, y)))
#define ROUND(a, b, c, d, e, f, g, h, Kt, Wt)                                  \
  {                                                                            \
    __m512i T1 = _mm512_add_epi32(                                             \
        h, _mm512_add_epi32(S1(e),                                             \
                              _mm512_add_epi32(Ch(e, f, g),                   \
                                                _mm512_add_epi32(Kt, Wt))));   \
    __m512i T2 = _mm512_add_epi32(S0(a), Maj(a, b, c));                        \
    h = g;                                                                     \
    g = f;                                                                     \
    f = e;                                                                     \
    e = _mm512_add_epi32(d, T1);                                               \
    d = c;                                                                     \
    c = b;                                                                     \
    b = a;                                                                     \
    a = _mm512_add_epi32(T1, T2);                                              \
  }

  for (int t = 0; t < 16; ++t) {
    ROUND(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[t]), W[t]);
  }

  for (int t = 16; t < 64; ++t) {
    __m512i newW = _mm512_add_epi32(
        _mm512_add_epi32(s1(W[(t - 2) & 0xf]), W[(t - 7) & 0xf]),
        _mm512_add_epi32(s0(W[(t - 15) & 0xf]), W[(t - 16) & 0xf]));
    W[t & 0xf] = newW;
    ROUND(a, b, c, d, e, f, g, h, _mm512_set1_epi32(K[t]), newW);
  }

  state[0] = _mm512_add_epi32(state[0], a);
  state[1] = _mm512_add_epi32(state[1], b);
  state[2] = _mm512_add_epi32(state[2], c);
  state[3] = _mm512_add_epi32(state[3], d);
  state[4] = _mm512_add_epi32(state[4], e);
  state[5] = _mm512_add_epi32(state[5], f);
  state[6] = _mm512_add_epi32(state[6], g);
  state[7] = _mm512_add_epi32(state[7], h);

#undef S0
#undef S1
#undef s0
#undef s1
#undef Ch
#undef Maj
#undef ROUND
}

}  // namespace

void sha256_avx512_16way(const uint8_t* const data[16], unsigned char* hash[16]) {
  ALIGN64 __m512i state[8];
  initialize(state);
  transform(state, data);

  ALIGN64 uint32_t digest[8][16];
  for (int i = 0; i < 8; ++i) {
    _mm512_store_si512(reinterpret_cast<__m512i*>(digest[i]), state[i]);
  }

  for (int lane = 0; lane < 16; ++lane) {
    for (int word = 0; word < 8; ++word) {
      uint32_t value = digest[word][lane];
#ifdef _MSC_VER
      value = _byteswap_ulong(value);
#else
      value = __builtin_bswap32(value);
#endif
      memcpy(hash[lane] + word * 4, &value, sizeof(uint32_t));
    }
  }
}
