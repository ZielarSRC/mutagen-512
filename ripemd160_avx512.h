#ifndef RIPEMD160_AVX512_H
#define RIPEMD160_AVX512_H

#include <cstdint>

namespace ripemd160_avx512 {

// Compute 16 RIPEMD-160 hashes in parallel using AVX-512 instructions.
// Each block pointer must reference a 64-byte buffer prepared with message
// and padding (the function mutates the buffers to append padding and length).
// The resulting 20-byte digests are written to the buffers supplied in the
// digests array.
void hash_32bytes_16way(uint8_t* blocks[16], unsigned char* digests[16]);

}  // namespace ripemd160_avx512

#endif  // RIPEMD160_AVX512_H
