from pathlib import Path

path = Path("v30/main.cpp")
text = path.read_text()

old_block = "constexpr size_t V30_GN_BLOCK_ROWS=8192;"
if text.count(old_block) != 1:
    raise SystemExit("V30 GN block constant not found")
text = text.replace(old_block, "constexpr size_t V30_GN_BLOCK_ROWS=256;", 1)

# The old one-row reducer becomes unused after the blocked reduction below.
old_helper = "static inline void accumulateGaussNewtonBasesRowV30(const double*x,const double*bases,\n                                                    V20DiagonalPartial&total){"
new_helper = "static inline __attribute__((unused)) void accumulateGaussNewtonBasesRowV30(const double*x,const double*bases,\n                                                    V20DiagonalPartial&total){"
if text.count(old_helper) != 1:
    raise SystemExit("one-row GN helper not found")
text = text.replace(old_helper, new_helper, 1)

old_reduced_tail = """        for(size_t rr=0;rr<count;rr++)
            accumulateGaussNewtonBasesRowV30(
                data.x.rowData(startRow+offset+rr),
                scratch.data()+rr*V30_GN_BASES_PER_ROW,total);"""
new_reduced_tail = """        for(size_t j=0;j<HIDDEN_NODES;j++){
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
                    const double*x=data.x.rowData(startRow+offset+rr);
                    const double jacobian=hiddenJacobianBase*x[k];
                    accumulator+=static_cast<long double>(jacobian)*jacobian;
                }
                total.values[k*HIDDEN_NODES+j]=accumulator;
            }
        }"""
if text.count(old_reduced_tail) != 1:
    raise SystemExit("reduced-scratch GN reduction tail not found")
text = text.replace(old_reduced_tail, new_reduced_tail, 1)

old_direct = """static void accumulateGaussNewtonSliceSimdV30(const Dataset&data,size_t firstRow,size_t lastRow,
                                              const Network&network,V20DiagonalPartial&out){
    alignas(64) double bases[V30_GN_BASES_PER_ROW];
    for(size_t row=firstRow;row<lastRow;row++){
        const double*x=data.x.rowData(row);
        gaussNewtonBasesRowV30(x,network,bases);
        accumulateGaussNewtonBasesRowV30(x,bases,out);
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
    raise SystemExit("direct GN reducer not found")
text = text.replace(old_direct, new_direct, 1)

path.write_text(text)
