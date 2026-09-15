from pathlib import Path

path = Path("v29/main.cpp")
text = path.read_text()
start = text.index('__attribute__((target("avx512f,avx512dq,fma"))) static void evaluateSliceV29(')
end = text.index('\nstatic V14Evaluation evaluateV29', start)
body = text[start:end]
needle = '    const __m512d zero=_mm512_setzero_pd();\n'
insert = (
    '    const __m512d zero=_mm512_setzero_pd();\n'
    '    const double* __restrict__ wOne=network.wOne.data();\n'
    '    const double* __restrict__ wTwo=network.wTwo.data();\n'
    '    const double* __restrict__ targets=data.y.data();\n'
    '    double* __restrict__ gradientData=out.gradient.data();\n'
)
if needle not in body:
    raise SystemExit('evaluateSliceV29 constant prologue not found')
body = body.replace(needle, insert, 1)
body = body.replace('network.wOne.data()', 'wOne')
body = body.replace('network.wTwo.data()', 'wTwo')
body = body.replace('data.y.data()', 'targets')
body = body.replace('out.gradient.data()', 'gradientData')
# Restore the initializer RHSs altered by the textual replacements.
body = body.replace('const double* __restrict__ wOne=wOne;', 'const double* __restrict__ wOne=network.wOne.data();')
body = body.replace('const double* __restrict__ wTwo=wTwo;', 'const double* __restrict__ wTwo=network.wTwo.data();')
body = body.replace('const double* __restrict__ targets=targets;', 'const double* __restrict__ targets=data.y.data();')
body = body.replace('double* __restrict__ gradientData=gradientData;', 'double* __restrict__ gradientData=out.gradient.data();')
path.write_text(text[:start] + body + text[end:])
