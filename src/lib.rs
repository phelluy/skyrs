/// Sparse matrix with skyline storage
/// # Examples
/// ```
/// use skyrs::Sky;
/// let coo = vec![
/// (0, 0, 2.),
/// (1, 1, 2.),
/// (2, 2, 2.),
/// (1, 0, -1.),
/// (0, 1, -1.),
/// (2, 1, -1.),
/// (1, 2, -1.),
/// ];
/// 
/// let mut sky = Sky::new(coo);
/// 
/// println!("Matrix full form:");
/// sky.print_coo();
/// 
/// let x0 = vec![1., 2., 3.];
/// let b = sky.vec_mult(&x0);
/// let x = sky.solve(b).unwrap();
/// 
/// println!("x0={:?}",x0);
/// println!("x={:?}",x);
/// 
/// let erreur: f64 = x0
/// .iter()
/// .zip(x.iter())
/// .map(|(&x0, &x)| (x - x0).abs())
/// .sum();
/// 
/// println!("error={:e}", erreur);
/// 
/// assert!(erreur < 1e-13);
/// ```
#[derive(Debug, Clone)]
pub struct Sky {
    coo: Vec<(usize, usize, f64)>,
    nrows: usize,
    ncols: usize,
    /// skyline
    sky: Vec<usize>,
    /// profile
    prof: Vec<usize>,
    /// rows of L
    ltab: Vec<Vec<f64>>,
    /// columns of U
    utab: Vec<Vec<f64>>,
}


const WIDTH: usize = 12;
const PREC: usize = 5;
const EXP: usize = 2;

const FMT: (usize, usize, usize) = (WIDTH, PREC, EXP);

/// Constant size formatter: useful for debug
fn fmt_f64(num: f64, fmt: (usize, usize, usize)) -> String {
    let width = fmt.0;
    let precision = fmt.1;
    let exp_pad = fmt.2;
    if num.is_finite() {
        let mut num = format!("{:+.precision$e}", num, precision = precision);
        // Safe to `unwrap` as `num` is guaranteed to contain `'e'`
        let exp = num.split_off(num.find('e').unwrap());

        let (sign, exp) = if exp.starts_with("e-") {
            ('-', &exp[2..])
        } else {
            ('+', &exp[1..])
        };
        num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

        format!("{:>width$}", num, width = width)
    } else {
        format!("{:>width$}", "Inf or NaN", width = width)
    }
}




impl Sky {
    pub fn new(coo: Vec<(usize, usize, f64)>) -> Sky {
        let imax = coo.iter().map(|(i, _, _)| i).max();
        let nrows = match imax {
            Some(i) => i + 1,
            None => 0,
        };
        let jmax = coo.iter().map(|(_, j, _)| j).max();
        let ncols = match jmax {
            Some(j) => j + 1,
            None => 0,
        };
        Sky {
            coo: coo,
            nrows: nrows,
            ncols: ncols,
            sky: vec![],
            prof: vec![],
            ltab: vec![vec![]; nrows],
            utab: vec![vec![]; ncols],
        }
    }

    /// Full print of the coo matrix
    #[allow(dead_code)]
    pub fn print_coo(&self) {
        // first search the size of the matrix
        if self.coo.len() == 0 {
            return;
        }

        let imax = self.coo.iter().map(|(i, _, _)| i).max().unwrap() + 1;
        let jmax = self.coo.iter().map(|(_, j, _)| j).max().unwrap() + 1;

        let mut full = vec![0.; imax * jmax];

        for (i, j, v) in self.coo.iter() {
            full[i * imax + j] = *v;
        }
        for i in 0..imax {
            for j in 0..jmax {
                let v = full[i * jmax + j];
                print!("{} ", fmt_f64(v, FMT));
            }
            println!("");
        }
    }

    /// Full print of the LU decomposition
    #[allow(dead_code)]
    pub fn print_lu(&self) {
        println!("L-I+U=");
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                let v = self.get_lu(i, j);
                print!("{} ", fmt_f64(v, FMT));
            }
            println!();
        }
    }

    /// Return the value at position (i,j) in L-I+U
    fn get_lu(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.nrows);
        assert!(j < self.ncols);
        if i > j && j >= self.prof[i] {
            self.get_l(i, j)
        } else if i <= j && i >= self.sky[j] {
            self.get_u(i, j)
        } else {
            0.
        }
    }

    /// Return the value at position (i,j) in L-I+U
    /// Fail if (i,j) is not in the profile or the skyline
    /// Used for debug
    #[allow(dead_code)]
    fn get_lu_try(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.nrows);
        assert!(j < self.ncols);
        if i > j && j >= self.prof[i] {
            self.get_l(i, j)
        } else if i <= j && i >= self.sky[j] {
            self.get_u(i, j)
        } else {
            panic!();
        }
    }

    /// Set the value (i,j) in L-I+U
    /// Fail if (i,j) is not in the skyline or in the profile
    fn set_lu(&mut self, i: usize, j: usize, v: f64) {
        if i > j {
            self.set_l(i, j, v);
        } else {
            self.set_u(i, j, v);
        }
    }

    /// Set the value (i,j) in L-I+U
    /// Do nothing if (i,j) is not in the skyline or in the profile
    #[allow(dead_code)]
    fn set_lu_try(&mut self, i: usize, j: usize, v: f64) {
        if i > j && j >= self.prof[i] {
            self.set_l(i, j, v);
        } else if i <= j && i >= self.sky[j] {
            self.set_u(i, j, v);
        } else {
            println!("i={} j={} is not in the pattern", i, j);
        }
    }

    /// Get the (i,j) value in L
    /// Fail if (i,j) is not in the profile
    /// or if i == j
    #[inline(always)] // probably useless, but...
    fn get_l(&self, i: usize, j: usize) -> f64 {
        self.ltab[i][j - self.prof[i]]
    }

    /// Get the (i,j) value in U
    /// Fail if (i,j) is not in the skyline
    #[inline(always)] // probably useless, but...
    fn get_u(&self, i: usize, j: usize) -> f64 {
        self.utab[j][i - self.sky[j]]
    }

    /// Set the (i,j) value in L
    /// Fail if (i,j) is not in the profile
    /// or if i == j
    #[inline(always)] // probably useless, but...
    fn set_l(&mut self, i: usize, j: usize, val: f64) {
        self.ltab[i][j - self.prof[i]] = val;
    }

    /// Set the (i,j) value in U
    /// Fail if (i,j) is not in the skyline
    #[inline(always)] // probably useless, but...
    fn set_u(&mut self, i: usize, j: usize, val: f64) {
        self.utab[j][i - self.sky[j]] = val;
    }

    /// Sort the coo array and combine values for the same (i,j) indices
    fn compress(&mut self) {
        if self.coo.is_empty() {
            return;
        };

        // lexicographic sorting for ensuring that identical entries
        // are near to each other
        self.coo
            .sort_by(|(i1, j1, _v1), (i2, j2, _v2)| (i1, j1).cmp(&(i2, j2)));

        let mut newcoo: Vec<(usize, usize, f64)> = vec![];

        let mut tr1 = self.coo[0];
        let mut cval = tr1.2;

        // combine successive identical entries in the new array
        self.coo
            .iter()
            .take(self.coo.len() - 1)
            .skip(1)
            .for_each(|tr2| {
                if (tr2.0, tr2.1) == (tr1.0, tr1.1) {
                    cval += tr2.2;
                } else {
                    newcoo.push((tr1.0, tr1.1, cval));
                    tr1 = *tr2;
                    cval = tr1.2;
                }
            });
        // last pushes
        let tr2 = self.coo.last().unwrap();
        if (tr2.0, tr2.1) == (tr1.0, tr1.1) {
            cval += tr2.2;
            newcoo.push((tr1.0, tr1.1, cval));
        } else {
            newcoo.push((tr1.0, tr1.1, cval));
            newcoo.push(*tr2);
        }

        self.coo = newcoo;
    }

    /// Matrix vector product using the coo array
    pub fn vec_mult(&self, u: &Vec<f64>) -> Vec<f64> {
        let mut v: Vec<f64> = vec![0.; self.nrows];
        if u.len() != self.ncols {
            panic!(
                "ncols={} is not equal to vector length={}",
                self.ncols,
                u.len()
            );
        };
        self.coo.iter().for_each(|(i, j, a)| {
            v[*i] += a * u[*j];
        });
        v
    }

    /// Convert the coo array to the skyline format internally
    /// The coo array is compressed before the construction
    fn coo_to_sky(&mut self) {
        assert_eq!(self.nrows, self.ncols);
        let n = self.nrows;
        if n == 0 {
            return;
        }
        let mut prof: Vec<usize> = (0..n).collect();
        let mut sky: Vec<usize> = (0..n).collect();
        self.compress();
        self.coo.iter().for_each(|(i, j, _v)| {
            if j > i {
                sky[*j] = sky[*j].min(*i);
            } else {
                prof[*i] = prof[*i].min(*j);
            }
        });
        self.prof = prof;
        self.sky = sky;

        // allocate space for the row of L
        // WITHOUT the diagonal terms
        for i in 0..n {
            self.ltab[i] = vec![0.; i - self.prof[i]];
        }
        // allocate space for the row of U
        // WITH the diagonal terms
        for j in 0..n {
            self.utab[j] = vec![0.; j - self.sky[j] + 1];
        }

        //println!("LU={:?}", self);

        for k in 0..self.coo.len() {
            let (i, j, v) = self.coo[k];
            self.set_lu(i, j, v);
        }
    }

    /// Performs a LU decomposition on the sparse matrix
    /// with the Doolittle algorithm
    /// Version with a for loop for debug
    #[allow(dead_code)]
    fn factolu_noscal(&mut self) -> Result<(), ()> {
        self.coo_to_sky();
        let n = self.nrows;
        for k in 1..n {
            for j in self.prof[k]..k {
                let mut lkj = self.get_l(k, j);
                let pmin = self.prof[k].max(self.sky[j]);
                for p in pmin..j {
                    lkj -= self.get_l(k, p) * self.get_u(p, j);
                }
                // the following line can produce NaN (but that's life...)
                lkj /= self.get_u(j, j);
                self.set_l(k, j, lkj);
            }
            for i in self.sky[k].max(1)..k + 1 {
                let mut uik = self.get_u(i, k);
                let pmin = self.prof[i].max(self.sky[k]);
                for p in pmin..i {
                    uik -= self.get_l(i, p) * self.get_u(p, k);
                }
                self.set_u(i, k, uik);
            }
        }
        Ok(())
    }

    /// Optimized scalar products of a sub-row of L
    /// with a sub-column of U (used in the L triangulation)
    #[inline(always)] // probably useless, but...
    fn scall(&self, i: usize, j: usize) -> f64 {
        let pmin = self.prof[i].max(self.sky[j]);
        let (lmin, lmax) = (pmin - self.prof[i], j - self.prof[i]);
        let (umin, umax) = (pmin - self.sky[j], j - self.sky[j]);
        let iiter = self.ltab[i][lmin..lmax].iter();
        let uiter = self.utab[j][umin..umax].iter();
        // todo: a good candidate for BLAS call
        let scal: f64 = iiter.zip(uiter).map(|(&l, &u)| l * u).sum();
        scal
    }

    /// Optimized scalar products of a sub-row of L
    /// with a sub-column of U (used in the U triangulation)
    #[inline(always)] // probably useless, but...
    fn scalu(&self, i: usize, j: usize) -> f64 {
        let pmin = self.prof[i].max(self.sky[j]);
        let (lmin, lmax) = (pmin - self.prof[i], i - self.prof[i]);
        let (umin, umax) = (pmin - self.sky[j], i - self.sky[j]);
        let iiter = self.ltab[i][lmin..lmax].iter();
        let uiter = self.utab[j][umin..umax].iter();
        // todo: a good candidate for BLAS call
        let scal: f64 = iiter.zip(uiter).map(|(&l, &u)| l * u).sum();
        scal
    }

    /// Performs a LU decomposition on the sparse matrix
    /// with the Doolittle algorithm
    pub fn factolu(&mut self) -> Result<(), String> {
        self.coo_to_sky();
        let n = self.nrows;
        for k in 1..n {
            for j in self.prof[k]..k {
                let mut lkj = self.get_l(k, j);
                lkj -= self.scall(k, j);
                // the following line may produce NaN (but we can live with this...)
                if self.get_u(j, j) == 0. {
                    return Err("A pivot is zero. Facto LU failed.".to_string());
                }
                lkj /= self.get_u(j, j);
                self.set_l(k, j, lkj);
            }
            for i in self.sky[k].max(1)..k + 1 {
                let mut uik = self.get_u(i, k);
                uik -= self.scalu(i, k);
                self.set_u(i, k, uik);
            }
        }
        Ok(())
    }
    /// Triangular solves
    /// Must be called after the LU decomposition !
    pub fn solve(&mut self, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
        let m = self.prof.len();
        if m == 0 {
            self.coo_to_sky();
            self.factolu()?;
        }
        // descente
        let n = self.nrows;
        for i in 0..n {
            for p in self.prof[i]..i {
                b[i] -= self.get_l(i, p) * b[p]; // self.ltab[i][p - self.prof[i]]
            }
        }
        // remontée
        b[n - 1] /= self.get_u(n - 1, n - 1);
        for j in (0..n - 1).rev() {
            for i in self.sky[j + 1]..j + 1 {
                //for i in 0..j {
                b[i] -= self.get_u(i, j + 1) * b[j + 1];
            }
            b[j] /= self.get_u(j, j);
        }
        Ok(b)
    }

    /// Performs a LU decomposition on the full matrix
    /// with the Doolittle algorithm
    /// Not efficient: used only for debug
    #[allow(dead_code)]
    fn factolu_full(&mut self) -> Result<(), ()> {
        self.coo_to_sky();
        let n = self.nrows;
        for k in 1..n {
            for j in 0..k {
                let mut lkj = self.get_lu(k, j);
                for p in 0..j {
                    lkj -= self.get_lu(k, p) * self.get_lu(p, j);
                }
                lkj /= self.get_lu(j, j);
                self.set_lu_try(k, j, lkj);
            }
            for i in 0..k + 1 {
                let mut uik = self.get_lu(i, k);
                for p in 0..i {
                    uik -= self.get_lu(i, p) * self.get_lu(p, k);
                }
                self.set_lu_try(i, k, uik);
            }
        }
        Ok(())
    }
    /// Triangular solves with the full structure
    /// Only here for debug
    #[allow(dead_code)]
    fn solve_slow(&self, mut b: Vec<f64>) -> Vec<f64> {
        // descente
        let n = self.nrows;
        for i in 0..n {
            for p in self.prof[i]..i {
                b[i] -= self.get_l(i, p) * b[p];
            }
        }
        b[n - 1] /= self.get_u(n - 1, n - 1);
        for i in (0..n - 1).rev() {
            for j in i + 1..n {
                b[i] -= self.get_u(i, j) * b[j];
            }
            b[i] /= self.get_u(i, i);
        }
        b
    }
    /// Perform the LU decomposition with the Gauss method
    /// on the full matrix
    /// Not efficient: only for debug purpose
    #[allow(dead_code)]
    fn factolu_gauss(&mut self) -> Result<(), ()> {
        self.coo_to_sky();

        let n = self.nrows;
        for p in 0..n - 1 {
            let piv = self.get_lu(p, p);
            // fill the column p in L
            for i in p + 1..n {
                let c = self.get_lu(i, p) / piv;
                for j in p + 1..n {
                    let mut aij = self.get_lu(i, j);
                    aij -= c * self.get_lu(p, j);
                    self.set_lu_try(i, j, aij);
                    self.set_lu_try(i, p, c);
                }
            }
        }
        Ok(())
    }
}

/// Inplace Gauss LU decomposition with row pivoting
/// on a full matrix. For debug purpose.
pub fn plu_facto(a: &mut Vec<Vec<f64>>, sigma: &mut Vec<usize>) {
    let n = a.len();
    a.iter().for_each(|row| assert_eq!(row.len(), n));
    assert_eq!(a.len(), n);
    // initialize permutation to identity
    for (p, sig) in sigma.iter_mut().enumerate() {
        *sig = p;
    }
    let n = sigma.len();
    // elimination in column p
    for p in 0..n - 1 {
        // search max pivot
        let mut sup = 0.;
        let mut p_max = p;
        for i in p..n {
            let abs_a_ip = (a[i][p]).abs();
            if sup < abs_a_ip {
                sup = abs_a_ip;
                p_max = i;
            }
        }

        // swap two lines
        for j in 0..n {
            let aux = a[p][j];
            a[p][j] = a[p_max][j];
            a[p_max][j] = aux;
        }

        // store permutation
        sigma.swap(p, p_max);

        for i in p + 1..n {
            let c = a[i][p] / a[p][p];
            for j in p + 1..n {
                a[i][j] -= c * a[p][j];
                a[i][p] = c;
            }
        }
    }
}

/// Inplace Doolittle LU decomposition on a full matrix
pub fn doolittle_lu(a: &mut Vec<Vec<f64>>) {
    let n = a.len();
    a.iter().for_each(|row| assert_eq!(row.len(), n));
    // pivot loop
    for k in 1..n {
        //update row left to the pivot
        for j in 0..k {
            for p in 0..j {
                a[k][j] -= a[k][p] * a[p][j];
            }
            a[k][j] /= a[j][j];
        }
        //update column p over the pivot
        for i in 1..k + 1 {
            for p in 0..i {
                a[i][k] -= a[i][p] * a[p][k];
            }
        }
    }
}

/// Permutation algorithm used by gauss_solve
fn gauss_permute(x: &mut Vec<f64>, sigma: &Vec<usize>) {
    let n = sigma.len();
    assert_eq!(n, x.len());
    for i in 0..n {
        let mut to_swap = sigma[i];
        while to_swap < i {
            to_swap = sigma[to_swap];
        }
        x.swap(i, to_swap);
    }
}

/// Triangular solves
/// plu_solve must be called first
/// and this has to be checked by the user before !
pub fn gauss_solve(a: &Vec<Vec<f64>>, sigma: &Vec<usize>, x: &mut Vec<f64>) {
    let n = a.len();
    assert_eq!(n, x.len());
    a.iter().for_each(|row| assert_eq!(row.len(), n));
    gauss_permute(x, sigma);

    for i in 1..n {
        for j in 0..i {
            x[i] -= a[i][j] * x[j];
        }
    }

    x[n - 1] = x[n - 1] / a[n - 1][n - 1];
    for i in (0..n - 1).rev() {
        for j in i + 1..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] = x[i] / a[i][i];
    }
}

// unit tests start

/// Test for small special matrices
#[test]
fn empty() {
    let coo: Vec<(usize, usize, f64)> = vec![];

    let mut sky = Sky::new(coo);
    sky.print_coo();

    sky.coo_to_sky();

    sky.factolu().unwrap();

    sky.print_lu();
}

#[test]
fn diagonal() {
    let n = 5;

    let coo = (0..n).map(|i| (i, i, 2.)).collect();

    let mut sky = Sky::new(coo);

    let u = (0..n).map(|i| i as f64).collect();

    let v = sky.vec_mult(&u);

    use float_eq::assert_float_eq;

    assert_float_eq!(
        v,
        (0..n).map(|i| (2 * i) as f64).collect(),
        abs_all <= 1e-12
    );

    let v = sky.solve(u).unwrap();
    println!("{:?}", v);

    //assert_float_eq!(v,(0..n).map(|i| i as f64 / 2.).collect(), abs_all <= 1e-12);
}

#[test]
fn small_matrix() {
    let mut coo: Vec<(usize, usize, f64)> = vec![];

    let n = 10;

    for i in 0..n {
        coo.push((i, i, 2.));
    }
    for i in 0..n - 1 {
        coo.push((i, i + 1, -1.));
        coo.push((i + 1, i, -1.));
    }

    // coo.push((0, 0, 0.));
    // coo.push((n - 1, n - 1, 0.));

    coo.push((3, 0, 0.1));
    coo.push((1, 4, 0.1));

    coo.push((4, 9, 0.1));

    // coo = vec![];

    // for i in 0..n {
    //     for j in 0..n {
    //         coo.push((i, j, 1. / (i + j + 1) as f64));
    //     }
    // }

    let mut sky = Sky::new(coo.clone());

    let u = vec![1.; n];

    let v1 = sky.vec_mult(&u);

    println!("Au={:?}", v1);

    sky.compress();

    let v2 = sky.vec_mult(&u);
    println!("Au={:?}", v2);

    v1.iter()
        .zip(v2.iter())
        .for_each(|(v1, v2)| assert!((*v1 - *v2).abs() < 1e-14));

    sky.coo_to_sky();

    sky.print_lu();

    sky.factolu().unwrap();

    println!("sky={:?}", sky);

    let mut a: Vec<Vec<f64>> = vec![vec![0. as f64; n]; n];
    //let mut sigma = vec![0; n];

    coo.iter().for_each(|(i, j, v)| {
        a[*i][*j] += *v;
    });

    // comparison with the full solver
    doolittle_lu(&mut a);

    let mut erreur = 0.;

    println!("LU sparse");
    sky.print_lu();

    println!("LU full");
    for i in 0..n {
        for j in 0..n {
            print!("{} ", fmt_f64(a[i][j], FMT));
            erreur += (a[i][j] - sky.get_lu(i, j)).abs();
        }
        println!();
    }
    println!("Facto error={}", fmt_f64(erreur, FMT));
    assert!(erreur < 1e-12);

    let x0 = (1..n + 1).map(|i| i as f64).collect();

    let b = sky.vec_mult(&x0);

    let x = sky.solve(b.clone()).unwrap();

    println!("x0={:?}", x0);
    println!("x={:?}", x);

    let erreur: f64 = x0
        .iter()
        .zip(x.iter())
        .map(|(&x0, &x)| (x - x0).abs())
        .sum();

    println!("erreur={}", fmt_f64(erreur, FMT));
    assert!(erreur < 1e-12);
}

// fn main() {
//     let coo = vec![
//         (0, 0, 2.),
//         (1, 1, 2.),
//         (2, 2, 2.),
//         (1, 0, -1.),
//         (0, 1, -1.),
//         (2, 1, -1.),
//         (1, 2, -1.),
//     ];

//     let mut sky = Sky::new(coo);

//     println!("Matrix full form:");
//     sky.print_coo();

//     let x0 = vec![1., 2., 3.];
//     let b = sky.vec_mult(&x0);
//     let x = sky.solve(b).unwrap();

//     println!("x0={:?}",x0);
//     println!("x={:?}",x);

//     let erreur: f64 = x0
//         .iter()
//         .zip(x.iter())
//         .map(|(&x0, &x)| (x - x0).abs())
//         .sum();

//     println!("error={:e}", erreur);

//     assert!(erreur < 1e-13);
// }