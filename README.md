# binary_matrix

[![CI](https://github.com/swenson/binary_matrix/actions/workflows/ci.yml/badge.svg)](https://github.com/swenson/binary_matrix/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/binary_matrix.svg)](https://crates.io/crates/binary_matrix)

Rust implementation of dense binary matrices and vectors.

Includes a SIMD implementation of a binary matrix.

## Rust versions and features

The crate builds on stable Rust 1.63 or newer with its default features.

| Feature | Description | Toolchain |
|---------|-------------|-----------|
| `rand`  | Random matrices and vectors | stable |
| `simd`  | `BinaryMatrixSimd`, built on `std::simd` | nightly |
| `bench` | `#[bench]` benchmarks | nightly |

Development uses the nightly pinned in `rust-toolchain.toml`:

```sh
cargo test --features rand,simd                # tests
cargo bench --features bench,rand,simd         # benchmarks
cargo +stable test --features rand             # stable subset
```

## TODO

* Arithmetic:
  * [ ] Implement the rest of the basic arithmetic between matrices and vectors
  * [ ] Faster matrix–vector multiplication using bits directly
  * [ ] Faster matrix–matrix multiplication using bits directly
  * [ ] Basic determinant calculation
  * [ ] Extract out basic reduced row echelon form into own method
  * [ ] Right multiplication of matrix by vector
* Kernel:
  * [ ] Use Lanczos algorithm
* Transpose:
  * [ ] Use rotates
  * [ ] Switch to SIMD
  * [ ] SIMD: Use `portable_simd`
  * [ ] Investigate using aarch64 assembly
  * [ ] Investigate using x86-64 assembly
* Implement row-centric matrix as well
* Sparse matrix support?

## License

[MIT](LICENSE.md)