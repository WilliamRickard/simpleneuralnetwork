from pathlib import Path

path = Path('v30/main.cpp')
text = path.read_text()
marker = '/**\n * Reduce one logical evaluator slice from precomputed forward states.'
if marker not in text:
    raise SystemExit('physical reducer marker not found')

macros = r'''
#define V30P_DECLARE_W1(G) \
    __m512d g00=_mm512_loadu_pd((G)+0*HIDDEN_NODES), g01=_mm512_loadu_pd((G)+0*HIDDEN_NODES+8); \
    __m512d g10=_mm512_loadu_pd((G)+1*HIDDEN_NODES), g11=_mm512_loadu_pd((G)+1*HIDDEN_NODES+8); \
    __m512d g20=_mm512_loadu_pd((G)+2*HIDDEN_NODES), g21=_mm512_loadu_pd((G)+2*HIDDEN_NODES+8); \
    __m512d g30=_mm512_loadu_pd((G)+3*HIDDEN_NODES), g31=_mm512_loadu_pd((G)+3*HIDDEN_NODES+8); \
    __m512d g40=_mm512_loadu_pd((G)+4*HIDDEN_NODES), g41=_mm512_loadu_pd((G)+4*HIDDEN_NODES+8); \
    __m512d g50=_mm512_loadu_pd((G)+5*HIDDEN_NODES), g51=_mm512_loadu_pd((G)+5*HIDDEN_NODES+8); \
    __m512d g60=_mm512_loadu_pd((G)+6*HIDDEN_NODES), g61=_mm512_loadu_pd((G)+6*HIDDEN_NODES+8); \
    __m512d g70=_mm512_loadu_pd((G)+7*HIDDEN_NODES), g71=_mm512_loadu_pd((G)+7*HIDDEN_NODES+8); \
    __m512d g80=_mm512_loadu_pd((G)+8*HIDDEN_NODES), g81=_mm512_loadu_pd((G)+8*HIDDEN_NODES+8); \
    __m512d g90=_mm512_loadu_pd((G)+9*HIDDEN_NODES), g91=_mm512_loadu_pd((G)+9*HIDDEN_NODES+8); \
    __m512d g100=_mm512_loadu_pd((G)+10*HIDDEN_NODES), g101=_mm512_loadu_pd((G)+10*HIDDEN_NODES+8)

#define V30P_ACCUM_W1(K,G0,G1) do { \
    const __m512d xv=_mm512_set1_pd(xq[(K)]); \
    (G0)=_mm512_fmadd_pd(xv,hiddenDeltaLow,(G0)); \
    (G1)=_mm512_fmadd_pd(xv,hiddenDeltaHigh,(G1)); \
} while(false)

#define V30P_ACCUM_ALL_W1() do { \
    V30P_ACCUM_W1(0,g00,g01); V30P_ACCUM_W1(1,g10,g11); V30P_ACCUM_W1(2,g20,g21); \
    V30P_ACCUM_W1(3,g30,g31); V30P_ACCUM_W1(4,g40,g41); V30P_ACCUM_W1(5,g50,g51); \
    V30P_ACCUM_W1(6,g60,g61); V30P_ACCUM_W1(7,g70,g71); V30P_ACCUM_W1(8,g80,g81); \
    V30P_ACCUM_W1(9,g90,g91); V30P_ACCUM_W1(10,g100,g101); \
} while(false)

#define V30P_STORE_W1(G) do { \
    _mm512_storeu_pd((G)+0*HIDDEN_NODES,g00); _mm512_storeu_pd((G)+0*HIDDEN_NODES+8,g01); \
    _mm512_storeu_pd((G)+1*HIDDEN_NODES,g10); _mm512_storeu_pd((G)+1*HIDDEN_NODES+8,g11); \
    _mm512_storeu_pd((G)+2*HIDDEN_NODES,g20); _mm512_storeu_pd((G)+2*HIDDEN_NODES+8,g21); \
    _mm512_storeu_pd((G)+3*HIDDEN_NODES,g30); _mm512_storeu_pd((G)+3*HIDDEN_NODES+8,g31); \
    _mm512_storeu_pd((G)+4*HIDDEN_NODES,g40); _mm512_storeu_pd((G)+4*HIDDEN_NODES+8,g41); \
    _mm512_storeu_pd((G)+5*HIDDEN_NODES,g50); _mm512_storeu_pd((G)+5*HIDDEN_NODES+8,g51); \
    _mm512_storeu_pd((G)+6*HIDDEN_NODES,g60); _mm512_storeu_pd((G)+6*HIDDEN_NODES+8,g61); \
    _mm512_storeu_pd((G)+7*HIDDEN_NODES,g70); _mm512_storeu_pd((G)+7*HIDDEN_NODES+8,g71); \
    _mm512_storeu_pd((G)+8*HIDDEN_NODES,g80); _mm512_storeu_pd((G)+8*HIDDEN_NODES+8,g81); \
    _mm512_storeu_pd((G)+9*HIDDEN_NODES,g90); _mm512_storeu_pd((G)+9*HIDDEN_NODES+8,g91); \
    _mm512_storeu_pd((G)+10*HIDDEN_NODES,g100); _mm512_storeu_pd((G)+10*HIDDEN_NODES+8,g101); \
} while(false)

'''

text = text.replace(marker, macros + marker, 1)
text = text.replace('V29_DECLARE_W1(gradient);', 'V30P_DECLARE_W1(gradient);', 1)
text = text.replace('V29_ACCUM_ALL_W1();', 'V30P_ACCUM_ALL_W1();', 1)
text = text.replace('V29_STORE_W1(gradient);', 'V30P_STORE_W1(gradient);', 1)

end_marker = '/**\n * Preserve v29\'s logical worker partition exactly while using up to twice as'
if end_marker not in text:
    raise SystemExit('physical evaluator marker not found')
undefs = '''#undef V30P_DECLARE_W1\n#undef V30P_ACCUM_W1\n#undef V30P_ACCUM_ALL_W1\n#undef V30P_STORE_W1\n\n'''
text = text.replace(end_marker, undefs + end_marker, 1)
path.write_text(text)
