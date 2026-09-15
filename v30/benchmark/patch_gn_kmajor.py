from pathlib import Path

path = Path("v30/benchmark/gn_candidate.cpp")
text = path.read_text()


def reorder(accumulator: str) -> None:
    global text
    old = f"""        for(size_t j=0;j<HIDDEN_NODES;j++){{
            const double outputJacobian=bases[j];
            {accumulator}.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
            const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
            for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){{
                const double jacobian=hiddenJacobianBase*x[k];
                {accumulator}.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }}
        }}"""
    new = f"""        for(size_t j=0;j<HIDDEN_NODES;j++){{
            const double outputJacobian=bases[j];
            {accumulator}.values[WONE_SIZE+j]+=static_cast<long double>(outputJacobian)*outputJacobian;
        }}
        for(size_t k=0;k<NUMBER_OF_VARIABLES;k++){{
            for(size_t j=0;j<HIDDEN_NODES;j++){{
                const double hiddenJacobianBase=bases[HIDDEN_NODES+j];
                const double jacobian=hiddenJacobianBase*x[k];
                {accumulator}.values[k*HIDDEN_NODES+j]+=static_cast<long double>(jacobian)*jacobian;
            }}
        }}"""
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected one {accumulator} GN reduction loop, found {count}")
    text = text.replace(old, new, 1)


reorder("total")
reorder("out")
path.write_text(text)
