from pathlib import Path

path = Path("v30/benchmark/gn_candidate.cpp")
text = path.read_text()

old = """        for(size_t j=0;j<HIDDEN_NODES;j++){
            const double outputJacobian=bases[j];
            total.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
            const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
                const double jacobian=hiddenJacobianBase*x[k];
                total.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }
        }"""

new = """        for(size_t j=0;j<HIDDEN_NODES;j++){
            const double outputJacobian=bases[j];
            total.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
        }
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){
            for(size_t j=0;j<HIDDEN_NODES;j++){
                const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
                const double jacobian=hiddenJacobianBase*x[k];
                total.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }
        }"""

count = text.count(old)
if count != 2:
    raise SystemExit(f"expected two GN reduction loops, found {count}")

path.write_text(text.replace(old, new))
