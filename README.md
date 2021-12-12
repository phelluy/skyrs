# skyrs
Skyline sparse matrix linear solver in Rust.

The skyline format is an old and rusty :-) way to solve sparse linear systems.

See for instance (among many other authors)

```bibtex
@book{dhatt1981presentation,
  title={Une pr{\'e}sentation de la m{\'e}thode des {\'e}l{\'e}ments finis},
  author={Dhatt, Gouri and Touzot, Gilbert},
  year={1981},
  publisher={Presses Universit{\'e} Laval}
}
```

A short description of the method is given at:



Do not forget to add the line
```
skyrs = {git = "https://github.com/phelluy/skyrs"}
```
in the `[dependencies]` section of your `Cargo.toml`.

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
-implement some utilities: refinement iterations for improved accuracy, add, multiply and conversion tools. 
