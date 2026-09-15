from pathlib import Path

path = Path("v29/main.cpp")
text = path.read_text()
begin = text.index("        /* Phase 1: first layer and hidden sigmoids, retaining v28 row arithmetic. */")
end = text.index("        /* Phase 2: output dot products with output weights live across the group. */", begin)

replacement = r'''        /* Experimental exact Phase 1: keep one 8-node half of W1 resident across the group. */
        {
            const __m512d w0=_mm512_loadu_pd(network.wOne.data()+0*HIDDEN_NODES);
            const __m512d w1=_mm512_loadu_pd(network.wOne.data()+1*HIDDEN_NODES);
            const __m512d w2=_mm512_loadu_pd(network.wOne.data()+2*HIDDEN_NODES);
            const __m512d w3=_mm512_loadu_pd(network.wOne.data()+3*HIDDEN_NODES);
            const __m512d w4=_mm512_loadu_pd(network.wOne.data()+4*HIDDEN_NODES);
            const __m512d w5=_mm512_loadu_pd(network.wOne.data()+5*HIDDEN_NODES);
            const __m512d w6=_mm512_loadu_pd(network.wOne.data()+6*HIDDEN_NODES);
            const __m512d w7=_mm512_loadu_pd(network.wOne.data()+7*HIDDEN_NODES);
            const __m512d w8=_mm512_loadu_pd(network.wOne.data()+8*HIDDEN_NODES);
            const __m512d w9=_mm512_loadu_pd(network.wOne.data()+9*HIDDEN_NODES);
            const __m512d w10=_mm512_loadu_pd(network.wOne.data()+10*HIDDEN_NODES);
            for(size_t tile=0;tile<tiles;tile++){
                const size_t tileRow=row+tile*V17_FORWARD_ROWS;
                const double*x[V17_FORWARD_ROWS];
                __m512d raw[V17_FORWARD_ROWS];
                for(size_t q=0;q<V17_FORWARD_ROWS;q++){x[q]=data.x.rowData(tileRow+q);raw[q]=zero;}
#define V30_HALF_ACC(K,W) do { for(size_t q=0;q<V17_FORWARD_ROWS;q++){ const __m512d v=_mm512_set1_pd(x[q][K]); raw[q]=_mm512_fmadd_pd(v,W,raw[q]); } } while(false)
                V30_HALF_ACC(0,w0); V30_HALF_ACC(1,w1); V30_HALF_ACC(2,w2); V30_HALF_ACC(3,w3);
                V30_HALF_ACC(4,w4); V30_HALF_ACC(5,w5); V30_HALF_ACC(6,w6); V30_HALF_ACC(7,w7);
                V30_HALF_ACC(8,w8); V30_HALF_ACC(9,w9); V30_HALF_ACC(10,w10);
#undef V30_HALF_ACC
                for(size_t q=0;q<V17_FORWARD_ROWS;q++)
                    _mm512_store_pd(hidden+(tile*V17_FORWARD_ROWS+q)*HIDDEN_NODES,raw[q]);
            }
        }
        {
            const __m512d w0=_mm512_loadu_pd(network.wOne.data()+0*HIDDEN_NODES+8);
            const __m512d w1=_mm512_loadu_pd(network.wOne.data()+1*HIDDEN_NODES+8);
            const __m512d w2=_mm512_loadu_pd(network.wOne.data()+2*HIDDEN_NODES+8);
            const __m512d w3=_mm512_loadu_pd(network.wOne.data()+3*HIDDEN_NODES+8);
            const __m512d w4=_mm512_loadu_pd(network.wOne.data()+4*HIDDEN_NODES+8);
            const __m512d w5=_mm512_loadu_pd(network.wOne.data()+5*HIDDEN_NODES+8);
            const __m512d w6=_mm512_loadu_pd(network.wOne.data()+6*HIDDEN_NODES+8);
            const __m512d w7=_mm512_loadu_pd(network.wOne.data()+7*HIDDEN_NODES+8);
            const __m512d w8=_mm512_loadu_pd(network.wOne.data()+8*HIDDEN_NODES+8);
            const __m512d w9=_mm512_loadu_pd(network.wOne.data()+9*HIDDEN_NODES+8);
            const __m512d w10=_mm512_loadu_pd(network.wOne.data()+10*HIDDEN_NODES+8);
            for(size_t tile=0;tile<tiles;tile++){
                const size_t tileRow=row+tile*V17_FORWARD_ROWS;
                const double*x[V17_FORWARD_ROWS];
                __m512d raw[V17_FORWARD_ROWS];
                for(size_t q=0;q<V17_FORWARD_ROWS;q++){x[q]=data.x.rowData(tileRow+q);raw[q]=zero;}
#define V30_HALF_ACC(K,W) do { for(size_t q=0;q<V17_FORWARD_ROWS;q++){ const __m512d v=_mm512_set1_pd(x[q][K]); raw[q]=_mm512_fmadd_pd(v,W,raw[q]); } } while(false)
                V30_HALF_ACC(0,w0); V30_HALF_ACC(1,w1); V30_HALF_ACC(2,w2); V30_HALF_ACC(3,w3);
                V30_HALF_ACC(4,w4); V30_HALF_ACC(5,w5); V30_HALF_ACC(6,w6); V30_HALF_ACC(7,w7);
                V30_HALF_ACC(8,w8); V30_HALF_ACC(9,w9); V30_HALF_ACC(10,w10);
#undef V30_HALF_ACC
                for(size_t q=0;q<V17_FORWARD_ROWS;q++)
                    _mm512_store_pd(hidden+(tile*V17_FORWARD_ROWS+q)*HIDDEN_NODES+8,raw[q]);
            }
        }
        for(size_t rr=0;rr<groupRows;rr++){
            double*h=hidden+rr*HIDDEN_NODES;
            _mm512_store_pd(h,sigmoidVectorV17(_mm512_load_pd(h)));
            _mm512_store_pd(h+8,sigmoidVectorV17(_mm512_load_pd(h+8)));
        }

'''

path.write_text(text[:begin] + replacement + text[end:])
