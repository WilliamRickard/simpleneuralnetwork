from pathlib import Path

path = Path("v30/benchmark/gn_candidate.cpp")
text = path.read_text()

reduced_output = """    for(size_t j=0;j<HIDDEN_NODES;j++){
        long double accumulator=total.values[WONE_SIZE+j];
        for(size_t rr=0;rr<count;rr++){
            const double outputJacobian=scratch[rr*V30_GN_BASES_PER_ROW+j];
            accumulator+=static_cast<long double>(outputJacobian)*outputJacobian;
        }
        total.values[WONE_SIZE+j]=accumulator;
    }"""
reduced_output4 = """    for(size_t jb=0;jb<HIDDEN_NODES;jb+=4){
        long double a0=total.values[WONE_SIZE+jb+0];
        long double a1=total.values[WONE_SIZE+jb+1];
        long double a2=total.values[WONE_SIZE+jb+2];
        long double a3=total.values[WONE_SIZE+jb+3];
        for(size_t rr=0;rr<count;rr++){
            const double*b=scratch+rr*V30_GN_BASES_PER_ROW+jb;
            a0+=static_cast<long double>(b[0])*b[0];
            a1+=static_cast<long double>(b[1])*b[1];
            a2+=static_cast<long double>(b[2])*b[2];
            a3+=static_cast<long double>(b[3])*b[3];
        }
        total.values[WONE_SIZE+jb+0]=a0;
        total.values[WONE_SIZE+jb+1]=a1;
        total.values[WONE_SIZE+jb+2]=a2;
        total.values[WONE_SIZE+jb+3]=a3;
    }"""

reduced_w1 = """    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
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
    }"""
reduced_w14 = """    for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
        for(size_t jb=0;jb<HIDDEN_NODES;jb+=4){
            long double a0=total.values[k*HIDDEN_NODES+jb+0];
            long double a1=total.values[k*HIDDEN_NODES+jb+1];
            long double a2=total.values[k*HIDDEN_NODES+jb+2];
            long double a3=total.values[k*HIDDEN_NODES+jb+3];
            for(size_t rr=0;rr<count;rr++){
                const double*x=data.x.rowData(startRow+rr);
                const double xv=x[k];
                const double*b=scratch+rr*V30_GN_BASES_PER_ROW+HIDDEN_NODES+jb;
                const double j0=b[0]*xv,j1=b[1]*xv,j2=b[2]*xv,j3=b[3]*xv;
                a0+=static_cast<long double>(j0)*j0;
                a1+=static_cast<long double>(j1)*j1;
                a2+=static_cast<long double>(j2)*j2;
                a3+=static_cast<long double>(j3)*j3;
            }
            total.values[k*HIDDEN_NODES+jb+0]=a0;
            total.values[k*HIDDEN_NODES+jb+1]=a1;
            total.values[k*HIDDEN_NODES+jb+2]=a2;
            total.values[k*HIDDEN_NODES+jb+3]=a3;
        }
    }"""

direct_output = reduced_output.replace("total.values", "out.values").replace("scratch[", "scratch[")
direct_output4 = reduced_output4.replace("total.values", "out.values").replace("scratch+", "scratch.data()+")
direct_w1 = reduced_w1.replace("total.values", "out.values").replace("startRow", "blockStart")
direct_w14 = reduced_w14.replace("total.values", "out.values").replace("startRow", "blockStart").replace("scratch+", "scratch.data()+")

for old,new,label in (
    (reduced_output,reduced_output4,"reduced output"),
    (reduced_w1,reduced_w14,"reduced W1"),
    (direct_output,direct_output4,"direct output"),
    (direct_w1,direct_w14,"direct W1"),
):
    count=text.count(old)
    if count != 1:
        raise SystemExit(f"{label} parameter-major loop not found: {count}")
    text=text.replace(old,new,1)

path.write_text(text)
