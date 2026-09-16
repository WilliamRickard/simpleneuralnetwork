#include <immintrin.h>

extern "C" __m256d _ZGVdN4v_exp(__m256d);

/**
 * Experimental link-time replacement for glibc's eight-lane vector exp.
 * The differential screen must establish bitwise identity before timing this path.
 */
extern "C" __attribute__((target("avx512f,avx512dq,avx2")))
__m512d __wrap__ZGVeN8v_exp(__m512d x){
    const __m256d low=_mm512_castpd512_pd256(x);
    const __m256d high=_mm512_extractf64x4_pd(x,1);
    const __m256d expLow=_ZGVdN4v_exp(low);
    const __m256d expHigh=_ZGVdN4v_exp(high);
    __m512d result=_mm512_castpd256_pd512(expLow);
    return _mm512_insertf64x4(result,expHigh,1);
}
