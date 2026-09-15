from pathlib import Path

path = Path("v30/benchmark/gn_candidate.cpp")
text = path.read_text()

old = """static void accumulateBasesBlockV30(const Dataset&data,size_t startRow,size_t count,
                                    const double*scratch,V20DiagonalPartial&total){
    for(size_t rr=0;rr<count;rr++){
        const double*x=data.x.rowData(startRow+rr);
        const double*bases=scratch+rr*V30_GN_BASES_PER_ROW;
        for(size_t j=0;j<HIDDEN_NODES;j++){
            const double outputJacobian=bases[j];
            total.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
            const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const double jacobian=hiddenJacobianBase*x[k];
                total.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }
        }
    }
}"""

new = """static void accumulateBasesBlockV30(const Dataset&data,size_t startRow,size_t count,
                                    const double*scratch,V20DiagonalPartial&total){
    for(size_t j=0;j<HIDDEN_NODES;j++){
        long double accumulator=total.values[WONE_SIZE+j];
        for(size_t rr=0;rr<count;rr++){
            const double outputJacobian=scratch[rr*V30_GN_BASES_PER_ROW+j];
            accumulator+=static_cast<long double>(outputJacobian)*outputJacobian;
        }
        total.values[WONE_SIZE+j]=accumulator;
    }
    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
        for(size_t j=0;j<HIDDEN_NODES;j++){
            long double accumulator=total.values[k*HIDDEN_NODES+j];
            for(size_t rr=0;rr<count;rr++){
                const double hiddenJacobianBase=
                    scratch[rr*V30_GN_BASES_PER_ROW+HIDDEN_NODES+j];
                const double*x=data.x.rowData(startRow+rr);
                const double jacobian=hiddenJacobianBase*x[k];
                accumulator+=static_cast<long double>(jacobian)*jacobian;
            }
            total.values[k*HIDDEN_NODES+j]=accumulator;
        }
    }
}"""

if text.count(old) != 1:
    raise SystemExit(f"expected one reduced-scratch GN reducer, found {text.count(old)}")

# A smaller block keeps the repeatedly scanned bases and inputs in private cache.
text = text.replace("constexpr size_t V30_GN_BLOCK_ROWS=8192;", "constexpr size_t V30_GN_BLOCK_ROWS=256;", 1)
text = text.replace(old, new, 1)
path.write_text(text)
