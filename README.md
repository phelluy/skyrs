# skyrs
Skyline sparse matrix linear solver in Rust. Parallelization thanks to 
a nested bisection renumbering.

The skyline format is an old and rusty :-) tool for solving sparse linear systems.

See for instance (among many others)

```bibtex
@book{dhatt2012finite,
  title={Finite element method},
  author={Dhatt, Gouri and Lefran{\c{c}}ois, Emmanuel and Touzot, Gilbert},
  year={2012},
  publisher={John Wiley \& Sons}
}
```

For a starting point to the nested bisection method, you can see:

```bibtex
@article{george1973nested,
  title={Nested dissection of a regular finite element mesh},
  author={George, Alan},
  journal={SIAM Journal on Numerical Analysis},
  volume={10},
  number={2},
  pages={345--363},
  year={1973},
  publisher={SIAM}
}
```

A short description of the method is given in the doc:

[https://github.com/phelluy/skyrs/blob/main/doc/skyline.pdf](https://github.com/phelluy/skyrs/blob/main/doc/skyline.pdf)

Do not forget to add the line
```
skyrs = {git = "https://github.com/phelluy/skyrs"}
```
in the `[dependencies]` section of your `Cargo.toml`. You also need a working installation of a BLAS 
(Basic Linear Algebra Subroutines) library on your system. For the moment, the `skyrs` library has been tested on

- Mac OS 12.1
- Linux
- Windows Subsystem for Linux (WSL) under MS Windows 10

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
 let b = sky.vec_mult(&x0);
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

## Comments

The library is working fine and can be used without too much worry.
However, this is a work in progress.
My todo list:

-change the implementation for handling other floating point types than f64 (f32 or complex types)

-implement the Cuthill-McKee algorithm for a better control of the fill-in

-optimisation: paralellism.

-implement some utilities: refinement iterations for improved accuracy, add, multiply and conversion tools. 
