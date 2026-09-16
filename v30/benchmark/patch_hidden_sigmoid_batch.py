from pathlib import Path

path = Path("v29/main.cpp")
text = path.read_text()

marker = '''/**
 * Evaluate one slice with exactly the same floating-point operations as v28,
'''
helper = r'''/**
 * Experimental v30 helper: end the 16 raw-vector live ranges before entering
 * libmvec.  Each lane receives exactly the same sub/exp/add/div sequence as
 * sigmoidVectorV17, in the same low/high order for every row.
 */
__attribute__((target("avx512f,avx512dq,fma"),noinline))
static void hiddenSigmoidBatchV30(const double*raw,double*hidden){
    const __m512d one=_mm512_set1_pd(1.0);
    const __m512d zero=_mm512_setzero_pd();
    for(size_t q=0;q<V17_FORWARD_ROWS;q++){
        const double*source=raw+q*HIDDEN_NODES;
        double*destination=hidden+q*HIDDEN_NODES;
        const __m512d low=_mm512_load_pd(source);
        const __m512d expLow=_ZGVeN8v_exp(_mm512_sub_pd(zero,low));
        _mm512_store_pd(destination,
                        _mm512_div_pd(one,_mm512_add_pd(one,expLow)));
        const __m512d high=_mm512_load_pd(source+8);
        const __m512d expHigh=_ZGVeN8v_exp(_mm512_sub_pd(zero,high));
        _mm512_store_pd(destination+8,
                        _mm512_div_pd(one,_mm512_add_pd(one,expHigh)));
    }
}

'''
if marker not in text:
    raise SystemExit("helper insertion marker not found")
text = text.replace(marker, helper + marker, 1)

old = '''            double*tileHidden=hidden+tile*V17_FORWARD_ROWS*HIDDEN_NODES;
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                double*h=tileHidden+q*HIDDEN_NODES;
                _mm512_store_pd(h,sigmoidVectorV17(rawLow[q]));
                _mm512_store_pd(h+8,sigmoidVectorV17(rawHigh[q]));
            }
'''
new = '''            alignas(64) double rawHidden[V17_FORWARD_ROWS*HIDDEN_NODES];
            for(size_t q=0;q<V17_FORWARD_ROWS;q++){
                double*r=rawHidden+q*HIDDEN_NODES;
                _mm512_store_pd(r,rawLow[q]);
                _mm512_store_pd(r+8,rawHigh[q]);
            }
            double*tileHidden=hidden+tile*V17_FORWARD_ROWS*HIDDEN_NODES;
            hiddenSigmoidBatchV30(rawHidden,tileHidden);
'''
if old not in text:
    raise SystemExit("phase-1 hidden sigmoid block not found")
text = text.replace(old, new, 1)
path.write_text(text)
print("patched", path)
