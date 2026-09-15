from pathlib import Path

path = Path("v29/main.cpp")
text = path.read_text()
begin = text.index("        /* Phase 4: one gradient state load/store for the complete group. */")
end = text.index("        row+=groupRows;", begin)
replacement = r'''        /* Experimental exact Phase 4: split independent W1 accumulators into 6+5 inputs. */
        double*gradient=out.gradient.data();
        const __m512d wTwoLow=_mm512_loadu_pd(network.wTwo.data());
        const __m512d wTwoHigh=_mm512_loadu_pd(network.wTwo.data()+8);

        /* Pass A: W2 plus W1 inputs 0..5. */
        __m512d gradientTwoLow=_mm512_loadu_pd(gradient+WONE_SIZE);
        __m512d gradientTwoHigh=_mm512_loadu_pd(gradient+WONE_SIZE+8);
        __m512d g00=_mm512_loadu_pd(gradient+0*HIDDEN_NODES),g01=_mm512_loadu_pd(gradient+0*HIDDEN_NODES+8);
        __m512d g10=_mm512_loadu_pd(gradient+1*HIDDEN_NODES),g11=_mm512_loadu_pd(gradient+1*HIDDEN_NODES+8);
        __m512d g20=_mm512_loadu_pd(gradient+2*HIDDEN_NODES),g21=_mm512_loadu_pd(gradient+2*HIDDEN_NODES+8);
        __m512d g30=_mm512_loadu_pd(gradient+3*HIDDEN_NODES),g31=_mm512_loadu_pd(gradient+3*HIDDEN_NODES+8);
        __m512d g40=_mm512_loadu_pd(gradient+4*HIDDEN_NODES),g41=_mm512_loadu_pd(gradient+4*HIDDEN_NODES+8);
        __m512d g50=_mm512_loadu_pd(gradient+5*HIDDEN_NODES),g51=_mm512_loadu_pd(gradient+5*HIDDEN_NODES+8);
        for(size_t rr=0;rr<groupRows;rr++){
            const double*h=hidden+rr*HIDDEN_NODES;
            const __m512d hiddenLow=_mm512_load_pd(h);
            const __m512d hiddenHigh=_mm512_load_pd(h+8);
            const __m512d d=_mm512_set1_pd(deltaThree[rr]);
            gradientTwoLow=_mm512_fmadd_pd(hiddenLow,d,gradientTwoLow);
            gradientTwoHigh=_mm512_fmadd_pd(hiddenHigh,d,gradientTwoHigh);
            const __m512d hiddenDeltaLow=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoLow),
                _mm512_mul_pd(hiddenLow,_mm512_sub_pd(one,hiddenLow)));
            const __m512d hiddenDeltaHigh=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoHigh),
                _mm512_mul_pd(hiddenHigh,_mm512_sub_pd(one,hiddenHigh)));
            const double*xq=data.x.rowData(row+rr);
#define V30_SPLIT_ACC(K,G0,G1) do { const __m512d xv=_mm512_set1_pd(xq[K]); G0=_mm512_fmadd_pd(xv,hiddenDeltaLow,G0); G1=_mm512_fmadd_pd(xv,hiddenDeltaHigh,G1); } while(false)
            V30_SPLIT_ACC(0,g00,g01); V30_SPLIT_ACC(1,g10,g11); V30_SPLIT_ACC(2,g20,g21);
            V30_SPLIT_ACC(3,g30,g31); V30_SPLIT_ACC(4,g40,g41); V30_SPLIT_ACC(5,g50,g51);
#undef V30_SPLIT_ACC
        }
        _mm512_storeu_pd(gradient+WONE_SIZE,gradientTwoLow);
        _mm512_storeu_pd(gradient+WONE_SIZE+8,gradientTwoHigh);
        _mm512_storeu_pd(gradient+0*HIDDEN_NODES,g00); _mm512_storeu_pd(gradient+0*HIDDEN_NODES+8,g01);
        _mm512_storeu_pd(gradient+1*HIDDEN_NODES,g10); _mm512_storeu_pd(gradient+1*HIDDEN_NODES+8,g11);
        _mm512_storeu_pd(gradient+2*HIDDEN_NODES,g20); _mm512_storeu_pd(gradient+2*HIDDEN_NODES+8,g21);
        _mm512_storeu_pd(gradient+3*HIDDEN_NODES,g30); _mm512_storeu_pd(gradient+3*HIDDEN_NODES+8,g31);
        _mm512_storeu_pd(gradient+4*HIDDEN_NODES,g40); _mm512_storeu_pd(gradient+4*HIDDEN_NODES+8,g41);
        _mm512_storeu_pd(gradient+5*HIDDEN_NODES,g50); _mm512_storeu_pd(gradient+5*HIDDEN_NODES+8,g51);

        /* Pass B: W1 inputs 6..10. Recompute only the cheap hidden derivative. */
        __m512d g60=_mm512_loadu_pd(gradient+6*HIDDEN_NODES),g61=_mm512_loadu_pd(gradient+6*HIDDEN_NODES+8);
        __m512d g70=_mm512_loadu_pd(gradient+7*HIDDEN_NODES),g71=_mm512_loadu_pd(gradient+7*HIDDEN_NODES+8);
        __m512d g80=_mm512_loadu_pd(gradient+8*HIDDEN_NODES),g81=_mm512_loadu_pd(gradient+8*HIDDEN_NODES+8);
        __m512d g90=_mm512_loadu_pd(gradient+9*HIDDEN_NODES),g91=_mm512_loadu_pd(gradient+9*HIDDEN_NODES+8);
        __m512d g100=_mm512_loadu_pd(gradient+10*HIDDEN_NODES),g101=_mm512_loadu_pd(gradient+10*HIDDEN_NODES+8);
        for(size_t rr=0;rr<groupRows;rr++){
            const double*h=hidden+rr*HIDDEN_NODES;
            const __m512d hiddenLow=_mm512_load_pd(h);
            const __m512d hiddenHigh=_mm512_load_pd(h+8);
            const __m512d d=_mm512_set1_pd(deltaThree[rr]);
            const __m512d hiddenDeltaLow=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoLow),
                _mm512_mul_pd(hiddenLow,_mm512_sub_pd(one,hiddenLow)));
            const __m512d hiddenDeltaHigh=_mm512_mul_pd(
                _mm512_mul_pd(d,wTwoHigh),
                _mm512_mul_pd(hiddenHigh,_mm512_sub_pd(one,hiddenHigh)));
            const double*xq=data.x.rowData(row+rr);
#define V30_SPLIT_ACC(K,G0,G1) do { const __m512d xv=_mm512_set1_pd(xq[K]); G0=_mm512_fmadd_pd(xv,hiddenDeltaLow,G0); G1=_mm512_fmadd_pd(xv,hiddenDeltaHigh,G1); } while(false)
            V30_SPLIT_ACC(6,g60,g61); V30_SPLIT_ACC(7,g70,g71); V30_SPLIT_ACC(8,g80,g81);
            V30_SPLIT_ACC(9,g90,g91); V30_SPLIT_ACC(10,g100,g101);
#undef V30_SPLIT_ACC
        }
        _mm512_storeu_pd(gradient+6*HIDDEN_NODES,g60); _mm512_storeu_pd(gradient+6*HIDDEN_NODES+8,g61);
        _mm512_storeu_pd(gradient+7*HIDDEN_NODES,g70); _mm512_storeu_pd(gradient+7*HIDDEN_NODES+8,g71);
        _mm512_storeu_pd(gradient+8*HIDDEN_NODES,g80); _mm512_storeu_pd(gradient+8*HIDDEN_NODES+8,g81);
        _mm512_storeu_pd(gradient+9*HIDDEN_NODES,g90); _mm512_storeu_pd(gradient+9*HIDDEN_NODES+8,g91);
        _mm512_storeu_pd(gradient+10*HIDDEN_NODES,g100); _mm512_storeu_pd(gradient+10*HIDDEN_NODES+8,g101);
'''
path.write_text(text[:begin] + replacement + text[end:])
