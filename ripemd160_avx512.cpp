#include "ripemd160_avx512.h"

#include <immintrin.h>
#include <cstring>
#include <cstdint>

namespace ripemd160_avx512 {

#ifdef _MSC_VER
#define ALIGN64 __declspec(align(64))
#else
#define ALIGN64 __attribute__((aligned(64)))
#endif

ALIGN64 static const uint32_t kInitialState[] = {
    // A
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    0x67452301ul, 0x67452301ul, 0x67452301ul, 0x67452301ul,
    // B
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul, 0xEFCDAB89ul,
    // C
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul, 0x98BADCFEul,
    // D
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    0x10325476ul, 0x10325476ul, 0x10325476ul, 0x10325476ul,
    // E
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul,
    0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul, 0xC3D2E1F0ul};

inline __m512i load_state(int index) {
  return _mm512_load_si512(reinterpret_cast<const __m512i*>(kInitialState + index * 16));
}

inline __m512i rol(const __m512i& x, int n) {
  return _mm512_or_si512(_mm512_slli_epi32(x, n), _mm512_srli_epi32(x, 32 - n));
}

inline __m512i f1(const __m512i& x, const __m512i& y, const __m512i& z) {
  return _mm512_xor_si512(x, _mm512_xor_si512(y, z));
}

inline __m512i f2(const __m512i& x, const __m512i& y, const __m512i& z) {
  return _mm512_or_si512(_mm512_and_si512(x, y), _mm512_andnot_si512(x, z));
}

inline __m512i mm512_not(const __m512i& x) {
  return _mm512_xor_si512(x, _mm512_set1_epi32(-1));
}

inline __m512i f3(const __m512i& x, const __m512i& y, const __m512i& z) {
  return _mm512_xor_si512(_mm512_or_si512(x, mm512_not(y)), z);
}

inline __m512i f4(const __m512i& x, const __m512i& y, const __m512i& z) {
  return _mm512_or_si512(_mm512_and_si512(x, z), _mm512_andnot_si512(z, y));
}

inline __m512i f5(const __m512i& x, const __m512i& y, const __m512i& z) {
  return _mm512_xor_si512(x, _mm512_or_si512(y, mm512_not(z)));
}

inline __m512i add3(const __m512i& a, const __m512i& b, const __m512i& c) {
  return _mm512_add_epi32(_mm512_add_epi32(a, b), c);
}

inline __m512i add4(const __m512i& a, const __m512i& b, const __m512i& c, const __m512i& d) {
  return _mm512_add_epi32(_mm512_add_epi32(a, b), _mm512_add_epi32(c, d));
}

inline __m512i load_word(uint8_t* const blocks[16], int index) {
  ALIGN64 uint32_t words[16];
  for (int lane = 0; lane < 16; ++lane) {
    words[lane] = reinterpret_cast<uint32_t*>(blocks[lane])[index];
  }
  return _mm512_load_si512(words);
}

inline void initialize(__m512i* state) {
  for (int i = 0; i < 5; ++i) {
    state[i] = load_state(i);
  }
}

inline void transform(__m512i* state, uint8_t* blocks[16]) {
  __m512i a1 = state[0];
  __m512i b1 = state[1];
  __m512i c1 = state[2];
  __m512i d1 = state[3];
  __m512i e1 = state[4];

  __m512i a2 = a1;
  __m512i b2 = b1;
  __m512i c2 = c1;
  __m512i d2 = d1;
  __m512i e2 = e1;

  __m512i u;
  ALIGN64 __m512i w[16];

  for (int i = 0; i < 16; ++i) {
    w[i] = load_word(blocks, i);
  }

#define ROUND(a, b, c, d, e, func, word, constant, rot)                         \
  u = add4(a, func(b, c, d), word, _mm512_set1_epi32(constant));               \
  a = _mm512_add_epi32(rol(u, rot), e);                                        \
  c = rol(c, 10)

#define R11(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f1, x, 0x00000000ul, r)
#define R21(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f2, x, 0x5A827999ul, r)
#define R31(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f3, x, 0x6ED9EBA1ul, r)
#define R41(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f4, x, 0x8F1BBCDCul, r)
#define R51(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f5, x, 0xA953FD4Eul, r)
#define R12(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f5, x, 0x50A28BE6ul, r)
#define R22(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f4, x, 0x5C4DD124ul, r)
#define R32(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f3, x, 0x6D703EF3ul, r)
#define R42(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f2, x, 0x7A6D76E9ul, r)
#define R52(a, b, c, d, e, x, r) ROUND(a, b, c, d, e, f1, x, 0x00000000ul, r)

  R11(a1, b1, c1, d1, e1, w[0], 11);
  R12(a2, b2, c2, d2, e2, w[5], 8);
  R11(e1, a1, b1, c1, d1, w[1], 14);
  R12(e2, a2, b2, c2, d2, w[14], 9);
  R11(d1, e1, a1, b1, c1, w[2], 15);
  R12(d2, e2, a2, b2, c2, w[7], 9);
  R11(c1, d1, e1, a1, b1, w[3], 12);
  R12(c2, d2, e2, a2, b2, w[0], 11);
  R11(b1, c1, d1, e1, a1, w[4], 5);
  R12(b2, c2, d2, e2, a2, w[9], 13);
  R11(a1, b1, c1, d1, e1, w[5], 8);
  R12(a2, b2, c2, d2, e2, w[2], 15);
  R11(e1, a1, b1, c1, d1, w[6], 7);
  R12(e2, a2, b2, c2, d2, w[11], 15);
  R11(d1, e1, a1, b1, c1, w[7], 9);
  R12(d2, e2, a2, b2, c2, w[4], 5);
  R11(c1, d1, e1, a1, b1, w[8], 11);
  R12(c2, d2, e2, a2, b2, w[13], 7);
  R11(b1, c1, d1, e1, a1, w[9], 13);
  R12(b2, c2, d2, e2, a2, w[6], 7);
  R11(a1, b1, c1, d1, e1, w[10], 14);
  R12(a2, b2, c2, d2, e2, w[15], 8);
  R11(e1, a1, b1, c1, d1, w[11], 15);
  R12(e2, a2, b2, c2, d2, w[8], 11);
  R11(d1, e1, a1, b1, c1, w[12], 6);
  R12(d2, e2, a2, b2, c2, w[1], 14);
  R11(c1, d1, e1, a1, b1, w[13], 7);
  R12(c2, d2, e2, a2, b2, w[10], 14);
  R11(b1, c1, d1, e1, a1, w[14], 9);
  R12(b2, c2, d2, e2, a2, w[3], 12);
  R11(a1, b1, c1, d1, e1, w[15], 8);
  R12(a2, b2, c2, d2, e2, w[12], 6);

  R21(e1, a1, b1, c1, d1, w[7], 7);
  R22(e2, a2, b2, c2, d2, w[6], 9);
  R21(d1, e1, a1, b1, c1, w[4], 6);
  R22(d2, e2, a2, b2, c2, w[11], 13);
  R21(c1, d1, e1, a1, b1, w[13], 8);
  R22(c2, d2, e2, a2, b2, w[3], 15);
  R21(b1, c1, d1, e1, a1, w[1], 13);
  R22(b2, c2, d2, e2, a2, w[7], 7);
  R21(a1, b1, c1, d1, e1, w[10], 11);
  R22(a2, b2, c2, d2, e2, w[0], 12);
  R21(e1, a1, b1, c1, d1, w[6], 9);
  R22(e2, a2, b2, c2, d2, w[13], 8);
  R21(d1, e1, a1, b1, c1, w[15], 7);
  R22(d2, e2, a2, b2, c2, w[5], 9);
  R21(c1, d1, e1, a1, b1, w[3], 15);
  R22(c2, d2, e2, a2, b2, w[10], 11);
  R21(b1, c1, d1, e1, a1, w[12], 7);
  R22(b2, c2, d2, e2, a2, w[14], 7);
  R21(a1, b1, c1, d1, e1, w[0], 12);
  R22(a2, b2, c2, d2, e2, w[15], 7);
  R21(e1, a1, b1, c1, d1, w[9], 15);
  R22(e2, a2, b2, c2, d2, w[8], 12);
  R21(d1, e1, a1, b1, c1, w[5], 9);
  R22(d2, e2, a2, b2, c2, w[12], 7);
  R21(c1, d1, e1, a1, b1, w[2], 11);
  R22(c2, d2, e2, a2, b2, w[4], 6);
  R21(b1, c1, d1, e1, a1, w[14], 7);
  R22(b2, c2, d2, e2, a2, w[9], 15);
  R21(a1, b1, c1, d1, e1, w[11], 13);
  R22(a2, b2, c2, d2, e2, w[1], 13);
  R21(e1, a1, b1, c1, d1, w[8], 12);
  R22(e2, a2, b2, c2, d2, w[2], 11);

  R31(d1, e1, a1, b1, c1, w[3], 11);
  R32(d2, e2, a2, b2, c2, w[15], 9);
  R31(c1, d1, e1, a1, b1, w[10], 13);
  R32(c2, d2, e2, a2, b2, w[5], 7);
  R31(b1, c1, d1, e1, a1, w[14], 6);
  R32(b2, c2, d2, e2, a2, w[1], 15);
  R31(a1, b1, c1, d1, e1, w[4], 7);
  R32(a2, b2, c2, d2, e2, w[3], 11);
  R31(e1, a1, b1, c1, d1, w[9], 14);
  R32(e2, a2, b2, c2, d2, w[7], 8);
  R31(d1, e1, a1, b1, c1, w[15], 9);
  R32(d2, e2, a2, b2, c2, w[14], 6);
  R31(c1, d1, e1, a1, b1, w[8], 13);
  R32(c2, d2, e2, a2, b2, w[6], 6);
  R31(b1, c1, d1, e1, a1, w[1], 15);
  R32(b2, c2, d2, e2, a2, w[9], 14);
  R31(a1, b1, c1, d1, e1, w[2], 14);
  R32(a2, b2, c2, d2, e2, w[11], 12);
  R31(e1, a1, b1, c1, d1, w[7], 8);
  R32(e2, a2, b2, c2, d2, w[8], 13);
  R31(d1, e1, a1, b1, c1, w[0], 13);
  R32(d2, e2, a2, b2, c2, w[12], 5);
  R31(c1, d1, e1, a1, b1, w[6], 6);
  R32(c2, d2, e2, a2, b2, w[2], 14);
  R31(b1, c1, d1, e1, a1, w[13], 5);
  R32(b2, c2, d2, e2, a2, w[10], 13);
  R31(a1, b1, c1, d1, e1, w[11], 12);
  R32(a2, b2, c2, d2, e2, w[0], 13);
  R31(e1, a1, b1, c1, d1, w[5], 7);
  R32(e2, a2, b2, c2, d2, w[4], 7);
  R31(d1, e1, a1, b1, c1, w[12], 5);
  R32(d2, e2, a2, b2, c2, w[13], 5);

  R41(c1, d1, e1, a1, b1, w[1], 11);
  R42(c2, d2, e2, a2, b2, w[8], 15);
  R41(b1, c1, d1, e1, a1, w[9], 12);
  R42(b2, c2, d2, e2, a2, w[6], 5);
  R41(a1, b1, c1, d1, e1, w[11], 14);
  R42(a2, b2, c2, d2, e2, w[4], 8);
  R41(e1, a1, b1, c1, d1, w[10], 15);
  R42(e2, a2, b2, c2, d2, w[1], 11);
  R41(d1, e1, a1, b1, c1, w[0], 14);
  R42(d2, e2, a2, b2, c2, w[3], 14);
  R41(c1, d1, e1, a1, b1, w[8], 15);
  R42(c2, d2, e2, a2, b2, w[11], 14);
  R41(b1, c1, d1, e1, a1, w[12], 9);
  R42(b2, c2, d2, e2, a2, w[15], 6);
  R41(a1, b1, c1, d1, e1, w[4], 8);
  R42(a2, b2, c2, d2, e2, w[0], 14);
  R41(e1, a1, b1, c1, d1, w[13], 9);
  R42(e2, a2, b2, c2, d2, w[5], 6);
  R41(d1, e1, a1, b1, c1, w[3], 14);
  R42(d2, e2, a2, b2, c2, w[12], 9);
  R41(c1, d1, e1, a1, b1, w[7], 5);
  R42(c2, d2, e2, a2, b2, w[2], 12);
  R41(b1, c1, d1, e1, a1, w[15], 6);
  R42(b2, c2, d2, e2, a2, w[13], 9);
  R41(a1, b1, c1, d1, e1, w[14], 8);
  R42(a2, b2, c2, d2, e2, w[9], 12);
  R41(e1, a1, b1, c1, d1, w[5], 6);
  R42(e2, a2, b2, c2, d2, w[7], 5);
  R41(d1, e1, a1, b1, c1, w[6], 5);
  R42(d2, e2, a2, b2, c2, w[10], 15);
  R41(c1, d1, e1, a1, b1, w[2], 12);
  R42(c2, d2, e2, a2, b2, w[14], 8);

  R51(b1, c1, d1, e1, a1, w[4], 9);
  R52(b2, c2, d2, e2, a2, w[12], 8);
  R51(a1, b1, c1, d1, e1, w[0], 15);
  R52(a2, b2, c2, d2, e2, w[15], 5);
  R51(e1, a1, b1, c1, d1, w[5], 5);
  R52(e2, a2, b2, c2, d2, w[10], 12);
  R51(d1, e1, a1, b1, c1, w[9], 11);
  R52(d2, e2, a2, b2, c2, w[4], 9);
  R51(c1, d1, e1, a1, b1, w[7], 6);
  R52(c2, d2, e2, a2, b2, w[1], 12);
  R51(b1, c1, d1, e1, a1, w[12], 8);
  R52(b2, c2, d2, e2, a2, w[5], 5);
  R51(a1, b1, c1, d1, e1, w[2], 13);
  R52(a2, b2, c2, d2, e2, w[8], 14);
  R51(e1, a1, b1, c1, d1, w[10], 12);
  R52(e2, a2, b2, c2, d2, w[7], 6);
  R51(d1, e1, a1, b1, c1, w[14], 5);
  R52(d2, e2, a2, b2, c2, w[6], 8);
  R51(c1, d1, e1, a1, b1, w[1], 12);
  R52(c2, d2, e2, a2, b2, w[2], 13);
  R51(b1, c1, d1, e1, a1, w[3], 13);
  R52(b2, c2, d2, e2, a2, w[13], 6);
  R51(a1, b1, c1, d1, e1, w[8], 14);
  R52(a2, b2, c2, d2, e2, w[14], 5);
  R51(e1, a1, b1, c1, d1, w[11], 11);
  R52(e2, a2, b2, c2, d2, w[0], 15);
  R51(d1, e1, a1, b1, c1, w[6], 8);
  R52(d2, e2, a2, b2, c2, w[3], 13);
  R51(c1, d1, e1, a1, b1, w[15], 5);
  R52(c2, d2, e2, a2, b2, w[9], 11);
  R51(b1, c1, d1, e1, a1, w[13], 6);
  R52(b2, c2, d2, e2, a2, w[11], 11);

  __m512i t = state[0];
  state[0] = add3(state[1], c1, d2);
  state[1] = add3(state[2], d1, e2);
  state[2] = add3(state[3], e1, a2);
  state[3] = add3(state[4], a1, b2);
  state[4] = add3(t, b1, c2);

#undef ROUND
#undef R11
#undef R21
#undef R31
#undef R41
#undef R51
#undef R12
#undef R22
#undef R32
#undef R42
#undef R52
}

void hash_32bytes_16way(uint8_t* blocks[16], unsigned char* digests[16]) {
  ALIGN64 __m512i state[5];
  initialize(state);

  static const uint64_t size_descriptor = 32ull << 3;
  static const unsigned char padding[64] = {0x80};

  for (int i = 0; i < 16; ++i) {
    std::memcpy(blocks[i] + 32, padding, 24);
    std::memcpy(blocks[i] + 56, &size_descriptor, 8);
  }

  transform(state, blocks);

  ALIGN64 uint32_t buffer[5][16];
  for (int i = 0; i < 5; ++i) {
    _mm512_store_si512(reinterpret_cast<__m512i*>(buffer[i]), state[i]);
  }

  for (int lane = 0; lane < 16; ++lane) {
    for (int word = 0; word < 5; ++word) {
      std::memcpy(digests[lane] + word * 4, &buffer[word][lane], sizeof(uint32_t));
    }
  }
}

}  // namespace ripemd160_avx512
