from pathlib import Path

path = Path("v30/benchmark/gn_candidate.cpp")
text = path.read_text()

# Keep the existing sub-50k parameter-major reduction with 256-row blocks.
old_block = "constexpr size_t V30_GN_BLOCK_ROWS=8192;"
if text.count(old_block) != 1:
    raise SystemExit("GN block constant not found")
text = text.replace(old_block, "constexpr size_t V30_GN_BLOCK_ROWS=256;", 1)

old_reduced = """static void accumulateBasesBlockV30(const Dataset&data,size_t startRow,size_t count,
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

new_reduced = """static void accumulateBasesBlockV30(const Dataset&data,size_t startRow,size_t count,
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
if text.count(old_reduced) != 1:
    raise SystemExit("reduced-scratch reducer not found")
text = text.replace(old_reduced, new_reduced, 1)

old_direct = """static void accumulateGaussNewtonSliceSimdV30(const Dataset&data,size_t firstRow,size_t lastRow,
                                              const Network&network,V20DiagonalPartial&out){
    alignas(64) double bases[V30_GN_BASES_PER_ROW];
    for(size_t row=firstRow;row<lastRow;row++){
        const double*x=data.x.rowData(row);
        gaussNewtonBasesRowV30(x,network,bases);
        for(size_t j=0;j<HIDDEN_NODES;j++){
            const double outputJacobian=bases[j];
            out.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
            const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const double jacobian=hiddenJacobianBase*x[k];
                out.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }
        }
    }
}"""

new_direct = """static void accumulateGaussNewtonSliceSimdV30(const Dataset&data,size_t firstRow,size_t lastRow,
                                              const Network&network,V20DiagonalPartial&out){
    vector<double>scratch(V30_GN_BLOCK_ROWS*V30_GN_BASES_PER_ROW);
    for(size_t blockStart=firstRow;blockStart<lastRow;blockStart+=V30_GN_BLOCK_ROWS){
        const size_t count=min(V30_GN_BLOCK_ROWS,lastRow-blockStart);
        for(size_t rr=0;rr<count;rr++)
            gaussNewtonBasesRowV30(data.x.rowData(blockStart+rr),network,
                                  scratch.data()+rr*V30_GN_BASES_PER_ROW);
        for(size_t j=0;j<HIDDEN_NODES;j++){
            long double accumulator=out.values[WONE_SIZE+j];
            for(size_t rr=0;rr<count;rr++){
                const double outputJacobian=scratch[rr*V30_GN_BASES_PER_ROW+j];
                accumulator+=static_cast<long double>(outputJacobian)*outputJacobian;
            }
            out.values[WONE_SIZE+j]=accumulator;
        }
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            for(size_t j=0;j<HIDDEN_NODES;j++){
                long double accumulator=out.values[k*HIDDEN_NODES+j];
                for(size_t rr=0;rr<count;rr++){
                    const double hiddenJacobianBase=
                        scratch[rr*V30_GN_BASES_PER_ROW+HIDDEN_NODES+j];
                    const double*x=data.x.rowData(blockStart+rr);
                    const double jacobian=hiddenJacobianBase*x[k];
                    accumulator+=static_cast<long double>(jacobian)*jacobian;
                }
                out.values[k*HIDDEN_NODES+j]=accumulator;
            }
        }
    }
}"""
if text.count(old_direct) != 1:
    raise SystemExit("direct SIMD reducer not found")
text = text.replace(old_direct, new_direct, 1)

path.write_text(text)
