import os


def _output_dir():
    return os.environ.get("INFINICORE_SPARSE_MTX_DIR")


def _fmt_density(density):
    return f"{density:.6g}".replace(".", "p")


def _write_header(f, rows, cols, nnz, *, name, density):
    f.write("%%MatrixMarket matrix coordinate real general\n")
    f.write(f"% name: {name}\n")
    if density is not None:
        f.write(f"% density: {density:.12g}\n")
    f.write(f"{rows} {cols} {nnz}\n")


def maybe_write_csr(name, rows, cols, crow, col, *, density=None):
    out_dir = _output_dir()
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir,
        f"{name}_rows{rows}_cols{cols}_nnz{len(col)}_density{_fmt_density(density or 0)}.mtx",
    )
    with open(path, "w", encoding="utf-8") as f:
        _write_header(f, rows, cols, len(col), name=name, density=density)
        for row in range(rows):
            for ptr in range(crow[row], crow[row + 1]):
                f.write(f"{row + 1} {col[ptr] + 1} 1\n")


def maybe_write_spvec(name, size, indices, *, density=None):
    out_dir = _output_dir()
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir,
        f"{name}_size{size}_nnz{len(indices)}_density{_fmt_density(density or 0)}.mtx",
    )
    with open(path, "w", encoding="utf-8") as f:
        _write_header(f, size, 1, len(indices), name=name, density=density)
        for index in indices:
            f.write(f"{index + 1} 1 1\n")
