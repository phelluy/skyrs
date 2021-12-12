# skyrs
Skyline sparse matrix linear solver

# How to use

A short example: 

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
