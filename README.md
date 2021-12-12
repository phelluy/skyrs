# skyrs
Skyline sparse matrix linear solver.

The skyline format is an old and rusty but robust way to solve sparse linear systems.

See for instance:

```bibtex
@book{dhatt1981presentation,
  title={Une pr{\'e}sentation de la m{\'e}thode des {\'e}l{\'e}ments finis},
  author={Dhatt, Gouri and Touzot, Gilbert},
  year={1981},
  publisher={Presses Universit{\'e} Laval}
}
```

# How to use

In this short example we first construct a skyline matrix A from its coordinate representation. We only give the non-zero entries as a list of `(i,j,v)` tuples such that
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
