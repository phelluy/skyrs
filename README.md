# skyrs
Skyline sparse matrix linear solver in Rust.

The skyline format is an old and rusty :-) tool for solving sparse linear systems.

A short description of the method is given in the doc:

[https://github.com/phelluy/skyrs/blob/main/doc/skyline.pdf](https://github.com/phelluy/skyrs/blob/main/doc/skyline.pdf)

Do not forget to add the line
```
skyrs = {git = "https://github.com/phelluy/skyrs"}
```
in the `[dependencies]` section of your `Cargo.toml`.

You also need a working installation of a BLAS 
(Basic Linear Algebra Subroutines) library on your system. For Linux or WSL in Windows 10
(with Ubuntu distribution), you can 
do:

```bash
sudo apt install libblas-dev
```

For the moment, the `skyrs` library has been tested on

- Mac OS 12.1
- Linux
- Windows Subsystem for Linux (WSL with Ubuntu) in MS Windows 10/11

## Example

 ```rust
 use skyrs::Sky;
 let coo = vec![
 (0, 0, 2.),
 (1, 1, 2.),
 (2, 2, 2.),
 (1, 0, -1.),
 (0, 1, -1.),
 (2, 1, -1.),
 (1, 2, -1.),
 ];
 
 let mut sky = Sky::new(coo);
 
 println!("Matrix full form:");
 sky.print_coo();
 
 let x0 = vec![1., 2., 3.];
 let b = sky.dot(&x0);
 let x = sky.solve(b).unwrap();
 
 println!("x0={:?}",x0);
 println!("x={:?}",x);
 
 let erreur: f64 = x0
 .iter()
 .zip(x.iter())
 .map(|(&x0, &x)| (x - x0).abs())
 .sum();
 
 println!("error={:e}", erreur);
 
 assert!(erreur < 1e-13);
 ```

Explanations: In this short example we first construct a skyline matrix A from its coordinate representation. We only give the non-zero entries as a list of `(i,j,v)` tuples such that
```
A[i][j] = v.
```  
We then display the matrix, compute a matrix-vector product
```
b = A x0,
```
and solve the linear system
```
A x = b
```
for checking that
```
x = x0
```
up to rounding errors.

## Visualization

You can visualize the sparsity pattern and values of the matrix using the `plot` method.
It generates a PNG heatmap where the pixel color represents the magnitude of the matrix coefficients (Black=0, Blue->Red=Low->High intensity).

```rust
// Plot the matrix to a 500x500 PNG image
// The file will be saved in "tmp_images/" with a timestamp
sky.plot("my_matrix_plot", 500).unwrap();
```

## Comments

The library is working fine and can be used without too much worry.

My todo list:

- change the implementation for handling other floating point types than f64 (f32 or complex types)

- optimisation: a better bisection, based for instance on spectral bisection. The parallelization, thanks to a nested bisection renumbering is implemented (but not activated by default).

- implement some utilities: refinement iterations for improved accuracy, add, multiply and conversion tools. 
