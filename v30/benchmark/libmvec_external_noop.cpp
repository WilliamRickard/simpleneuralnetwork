#include <immintrin.h>

extern "C" __attribute__((noinline,noclone,target("avx512f,avx512dq"),visibility("default")))
__m512d v30_external_vec_noop(__m512d x){
    asm volatile("" : "+v"(x) : : "memory");
    return x;
}
