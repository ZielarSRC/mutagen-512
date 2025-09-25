#ifndef SHA256_AVX512_H
#define SHA256_AVX512_H

#include <cstdint>

// Compute 16 SHA-256 hashes in parallel using AVX-512 instructions.
// Each entry in the data array must point to a 64-byte message block that already
// contains the SHA-256 padding and length encoding.
// The resulting 32-byte digests are stored in the buffers referenced by the hash array.
void sha256_avx512_16way(const uint8_t* const data[16], unsigned char* hash[16]);

#endif  // SHA256_AVX512_H
