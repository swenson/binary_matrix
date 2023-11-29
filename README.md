# binary_matrix

Rust implementation of dense binary matrices and vectors.

Includes a SIMD implementation of a binary matrix.

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