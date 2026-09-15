from pathlib import Path

path = Path("v30/benchmark/gn_candidate.cpp")
text = path.read_text()

old_output = """        for(size_t j=0;j<HIDDEN_NODES;j++){
            long double accumulator=out.values[WONE_SIZE+j];
            for(size_t rr=0;rr<count;rr++){
                const double outputJacobian=scratch[rr*V30_GN_BASES_PER_ROW+j];
                accumulator+=static_cast<long double>(outputJacobian)*outputJacobian;
            }
            out.values[WONE_SIZE+j]=accumulator;
        }"""
new_output = """        for(size_t jb=0;jb<HIDDEN_NODES;jb+=4){
            long double a0=out.values[WONE_SIZE+jb+0];
            long double a1=out.values[WONE_SIZE+jb+1];
            long double a2=out.values[WONE_SIZE+jb+2];
            long double a3=out.values[WONE_SIZE+jb+3];
            for(size_t rr=0;rr<count;rr++){
                const double*b=scratch.data()+rr*V30_GN_BASES_PER_ROW+jb;
                a0+=static_cast<long double>(b[0])*b[0];
                a1+=static_cast<long double>(b[1])*b[1];
                a2+=static_cast<long double>(b[2])*b[2];
                a3+=static_cast<long double>(b[3])*b[3];
            }
            out.values[WONE_SIZE+jb+0]=a0;
            out.values[WONE_SIZE+jb+1]=a1;
            out.values[WONE_SIZE+jb+2]=a2;
            out.values[WONE_SIZE+jb+3]=a3;
        }"""

old_w1 = """        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
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
        }"""
new_w1 = """        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            for(size_t jb=0;jb<HIDDEN_NODES;jb+=4){
                long double a0=out.values[k*HIDDEN_NODES+jb+0];
                long double a1=out.values[k*HIDDEN_NODES+jb+1];
                long double a2=out.values[k*HIDDEN_NODES+jb+2];
                long double a3=out.values[k*HIDDEN_NODES+jb+3];
                for(size_t rr=0;rr<count;rr++){
                    const double*x=data.x.rowData(blockStart+rr);
                    const double xv=x[k];
                    const double*b=scratch.data()+rr*V30_GN_BASES_PER_ROW+HIDDEN_NODES+jb;
                    const double j0=b[0]*xv,j1=b[1]*xv,j2=b[2]*xv,j3=b[3]*xv;
                    a0+=static_cast<long double>(j0)*j0;
                    a1+=static_cast<long double>(j1)*j1;
                    a2+=static_cast<long double>(j2)*j2;
                    a3+=static_cast<long double>(j3)*j3;
                }
                out.values[k*HIDDEN_NODES+jb+0]=a0;
                out.values[k*HIDDEN_NODES+jb+1]=a1;
                out.values[k*HIDDEN_NODES+jb+2]=a2;
                out.values[k*HIDDEN_NODES+jb+3]=a3;
            }
        }"""

# Reduced-scratch variant uses total and offset rather than out/blockStart.
old_output_total = old_output.replace("out.values", "total.values").replace("scratch.data()", "scratch")
new_output_total = new_output.replace("out.values", "total.values").replace("scratch.data()", "scratch")
old_w1_total = old_w1.replace("out.values", "total.values").replace("blockStart", "startRow").replace("scratch.data()", "scratch")
# The reduced reducer's start is already the current block's absolute startRow in gn_candidate.
new_w1_total = new_w1.replace("out.values", "total.values").replace("blockStart", "startRow").replace("scratch.data()", "scratch")

replacements = [
    (old_output_total, new_output_total, "reduced output"),
    (old_w1_total, new_w1_total, "reduced W1"),
    (old_output, new_output, "direct output"),
    (old_w1, new_w1, "direct W1"),
]
for old,new,label in replacements:
    if text.count(old) != 1:
        raise SystemExit(f"{label} parameter-major loop not found: {text.count(old)}")
    text=text.replace(old,new,1)

path.write_text(text)
