const WIDTH: usize = 12;
const PREC: usize = 5;
const EXP: usize = 2;

const FMT: (usize, usize, usize) = (WIDTH, PREC, EXP);

fn fmt_f64(num: f64, fmt: (usize, usize, usize)) -> String {
    let width = fmt.0;
    let precision = fmt.1;
    let exp_pad = fmt.2;
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
}

/// Sparse matrix with skyline storage 
#[derive(Debug)]
struct SkyMat {
    coo: Vec<(usize, usize, f64)>,
    nrows: usize,
    ncols: usize,
    vkgd: Vec<f64>,
    vkgs: Vec<f64>,
    vkgi: Vec<f64>,
    skld: Vec<usize>,
    ikld: Vec<usize>,
    /// skyline
    sky: Vec<usize>,    
    /// profile
    prof: Vec<usize>,    
    /// rows of L
    ltab: Vec<Vec<f64>>,   
    /// columns of U
    utab: Vec<Vec<f64>>,   
}

impl SkyMat {
    fn new(coo: Vec<(usize, usize, f64)>) -> SkyMat {
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
        SkyMat {
            coo: coo,
            nrows: nrows,
            ncols: ncols,
            vkgd: vec![],
            vkgs: vec![],
            vkgi: vec![],
            skld: vec![],
            ikld: vec![],
            sky: vec![],
            prof: vec![],
            ltab: vec![vec![];nrows],
            utab: vec![vec![];ncols]
        }
    }

    fn print(&self) {
        // first search the size of the matrix
        let imax = self.coo.iter().map(|(i, _, _)| i).max().unwrap() + 1;
        let jmax = self.coo.iter().map(|(_, j, _)| j).max().unwrap() + 1;

        let mut full = vec![0.; imax * jmax];

        for (i, j, v) in self.coo.iter() {
            full[i * imax + j] = *v;
        }
        println!("A=");
        for i in 0..imax {
            for j in 0..jmax {
                let v = full[i * jmax + j];
                print!("{} ", fmt_f64(v, FMT));
            }
            println!("");
        }
    }

    /// full print of the LU decomposition
    fn print_lu(&self) {
        println!("L-I+U=");
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                let v = self.get(i, j);
                print!("{} ", fmt_f64(v, FMT));
            }
            println!();
        }
    }

    fn get(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.nrows);
        assert!(j < self.ncols);
        if i > j {
            self.get_l(i, j)
        } else {
            self.get_u(i, j)
        }
    }

    fn set(&mut self, i: usize, j: usize, val: f64) -> Result<(), ()> {
        set_sky(
            i,
            j,
            val,
            &mut self.vkgd,
            &mut self.vkgs,
            &mut self.vkgi,
            &self.skld,
            &self.ikld,
        )
    }

    fn get_l(&self, i: usize, j: usize) -> f64 {
        // mémo:
        // self.vkgs[self.skld[j] + j - i - 1]   U(i,j)
        // self.vkgi[self.ikld[i] + i - j - 1]   L(i,j)
        self.vkgi[self.ikld[i] + i - j - 1]
    }

    fn get_u(&self, i: usize, j: usize) -> f64 {
        if i != j {
            self.vkgs[self.skld[j] + j - i - 1]
        } else {
            self.vkgd[i]
        }
    }

    fn set_l(&mut self, i: usize, j: usize, val: f64) -> Result<(), ()> {
        // mémo:
        // self.vkgs[self.skld[j] + j - i - 1]   U(i,j)
        // self.vkgi[self.ikld[i] + i - j - 1]   L(i,j)
        self.vkgi[self.ikld[i] + i - j - 1] = val;
        Ok(())
    }

    fn set_u(&mut self, i: usize, j: usize, val: f64) -> Result<(), ()> {
        if i != j {
            self.vkgs[self.skld[j] + j - i - 1] = val;
        } else {
            self.vkgd[i] = val;
        }
        Ok(())
    }

    /// sort coo array and combine values for the same (i,j) index
    fn compress(&mut self) {
        if self.coo.is_empty() {
            return;
        };

        // lexicographic sorting for ensuring that identical entries
        // are near to each other
        self.coo
            .sort_by(|(i1, j1, _v1), (i2, j2, _v2)| (i1, j1).cmp(&(i2, j2)));
        //println!("{:?}", vtuple);

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
        // last push
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

    fn vec_mult(&self, u: &Vec<f64>) -> Vec<f64> {
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

    fn coo_to_sky(&mut self) {
        assert_eq!(self.nrows, self.ncols);
        let n = self.nrows;
        if n == 0 {
            return;
        }
        let mut prof = vec![0; n];
        let mut sky = vec![0; n];
        // initially prof and sky count the non zero terms
        self.coo.iter().for_each(|(i, j, _v)| {
            if j > i {
                sky[*j] = sky[*j].max(j - i);
            } else {
                prof[*i] = prof[*i].max(i - j);
            }
        });
        self.prof = prof;
        self.sky = sky;
        self.skld = vec![0; n + 1];
        for i in 0..n {
            self.skld[i + 1] = self.skld[i] + self.sky[i];
            self.sky[i] = i - self.sky[i];
        }
        self.ikld = vec![0; n + 1];
        for i in 0..n {
            self.ikld[i + 1] = self.ikld[i] + self.prof[i];
            self.prof[i] = i - self.prof[i];
        }
        // now sky and prof contain the true profile and skyline
        // stored values in the upper and lower triangles
        let snnz = self.skld[n];
        let innz = self.ikld[n];
        let mut vkgd = vec![0.; n];
        let mut vkgs = vec![0.; snnz];
        let mut vkgi = vec![0.; innz];

        self.coo.iter().for_each(|(i, j, v)| {
            set_sky(
                *i, *j, *v, &mut vkgd, &mut vkgs, &mut vkgi, &self.skld, &self.ikld,
            )
            .unwrap();
        });
        self.vkgd = vkgd;
        self.vkgs = vkgs;
        self.vkgi = vkgi;

        // new format
        // allocate space for the row of L
        // without the diagonal terms
        for i in 0..n {
            self.ltab[i] = vec![0.; i-self.prof[i]];
    }
        // allocate space for the row of U
        // with the diagonal terms
        for j in 0..n {
            self.utab[j] = vec![0.; j-self.sky[j]+1];
    }

    }

    fn factolu_full(&mut self) -> Result<(), ()> {
        self.coo_to_sky();
        let n = self.nrows;
        for k in 1..n {
            for j in 0..k {
                let mut lkj = self.get(k, j);
                for p in 0..j {
                    lkj -= self.get(k, p) * self.get(p, j);
                }
                lkj /= self.get(j, j);
                self.set(k, j, lkj)?;
            }
            for i in 0..k + 1 {
                let mut uik = self.get(i, k);
                for p in 0..i {
                    uik -= self.get(i, p) * self.get(p, k);
                }
                self.set(i, k, uik)?;
            }
        }
        Ok(())
    }

    fn factolu(&mut self) -> Result<(), ()> {
        self.coo_to_sky();
        let n = self.nrows;
        for k in 1..n {
            for j in self.prof[k]..k {
                let mut lkj = self.get_l(k, j);
                let pmin = self.prof[k].max(self.sky[j]);
                assert!(pmin <= j);
                for p in pmin..j {
                    lkj -= self.get_l(k, p) * self.get_u(p, j);
                }
                lkj /= self.get_u(j, j); // warning if div by zero !
                self.set_l(k, j, lkj)?;
            }
            for i in self.sky[k]..k + 1 {
                let mut uik = self.get(i, k);
                let pmin = self.prof[i].max(self.sky[k]);
                for p in pmin..i {
                    uik -= self.get_l(i, p) * self.get_u(p, k);
                }
                self.set_u(i, k, uik)?;
            }
        }
        Ok(())
    }

    fn solve(&self, mut b: Vec<f64>) -> Vec<f64> {
        // descente
        let n = self.nrows;
        for i in 0..n {
            for p in self.prof[i]..i {
                b[i] -= self.get_l(i, p) * b[p];
            }
        }
        b[n - 1] /= self.get(n - 1, n - 1);
        for j in (0..n - 1).rev() {
            for i in self.sky[j + 1]..j + 1 {
                //for i in 0..j {
                b[i] -= self.get_u(i, j + 1) * b[j + 1];
            }
            b[j] /= self.get_u(j, j);
        }
        b
    }

    fn solve_slow(&self, mut b: Vec<f64>) -> Vec<f64> {
        // descente
        let n = self.nrows;
        for i in 0..n {
            for p in self.prof[i]..i {
                b[i] -= self.get(i, p) * b[p];
            }
        }
        b[n - 1] /= self.get(n - 1, n - 1);
        for i in (0..n - 1).rev() {
            for j in i + 1..n {
                b[i] -= self.get(i, j) * b[j];
            }
            b[i] /= self.get(i, i);
        }
        b
    }

    fn factolu_classic(&mut self) -> Result<(), ()> {
        self.coo_to_sky();

        let n = self.skld.len() - 1;
        for p in 0..n - 1 {
            let piv = self.vkgd[p];
            // fill the column p in L
            for i in p + 1..n {
                let c = self.get(i, p) / piv; // c = a[i,p] /  a[p,p]
                self.set(i, p, c)?;
            }
            for i in p + 1..n {
                // use the column p of L for elimination
                // Ui = Ui - c Up   U[i,j] = U[i,j] - c U[p,j] for j >= i
                // diagonal term
                for j in i..n {
                    let mut u = self.get(i, j);
                    let c = self.get(i, p);
                    u -= c * self.get(p, j);
                    self.set(i, j, u)?;
                }
                // upper diagonal term
            }
        }
        Ok(())
    }
}

// /// Display coo matrix in full
// fn print_sky(
//     vkgd: &Vec<f64>,
//     vkgs: &Vec<f64>,
//     vkgi: &Vec<f64>,
//     skld: &Vec<usize>,
//     ikld: &Vec<usize>,
// ) {
//     // first search the size of the matrix
//     let n = skld.len() - 1;
//     println!("nrows={}", n);
//     println!("ncols={}", n);

//     println!("full_lu=");
//     for i in 0..n {
//         for j in 0..n {
//             print!(
//                 "{} ",
//                 fmt_f64(get_sky(i, j, vkgd, vkgs, vkgi, skld, ikld), FMT)
//             );
//         }
//         println!("");
//     }
//     //println!("full={:?}", full);
// }

/// Convert a coo matrix to compressed sparse row (csr) format
/// the matrix must be first sorted and compressed !
#[allow(dead_code)]
fn coo_to_csr(
    val_coo: Vec<f64>,
    iv: Vec<usize>,
    jv: Vec<usize>,
) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    // some checks
    assert_eq!(val_coo.len(), iv.len());
    assert_eq!(val_coo.len(), jv.len());
    let nrows = *iv.iter().max().unwrap() + 1;
    // first count non zero values in each row
    let mut row_count = vec![0; nrows];
    iv.iter().for_each(|iv| row_count[*iv] += 1);
    println!("{:?}", row_count);
    let mut row_start = vec![0; nrows + 1];
    for i in 0..nrows {
        row_start[i + 1] = row_start[i] + row_count[i];
    }
    let nnz = row_start[nrows];
    println!("{:?} nnz={}", row_start, nnz);
    (val_coo, row_start, jv)
}

fn get_sky(
    i: usize,
    j: usize,
    vkgd: &Vec<f64>,
    vkgs: &Vec<f64>,
    vkgi: &Vec<f64>,
    skld: &Vec<usize>,
    ikld: &Vec<usize>,
) -> f64 {
    let n = vkgd.len();
    assert!(i < n);
    assert!(j < n);
    if i == j {
        //println!("(i,j)={} {}",i,j);
        vkgd[i]
    } else if j > i && j - i <= skld[j + 1] - skld[j] {
        let k = skld[j] + j - i - 1;
        vkgs[k]
    } else if j < i && i - j <= ikld[i + 1] - ikld[i] {
        let k = ikld[i] + i - j - 1;
        vkgi[k]
    } else {
        //println!("Read out of profile {} {}",i,j);
        0.
    }
}

//use std::error::Error;

fn set_sky(
    i: usize,
    j: usize,
    val: f64,
    vkgd: &mut Vec<f64>,
    vkgs: &mut Vec<f64>,
    vkgi: &mut Vec<f64>,
    skld: &Vec<usize>,
    ikld: &Vec<usize>,
) -> Result<(), ()> {
    let n = vkgd.len();
    assert!(i < n);
    assert!(j < n);

    if i == j {
        vkgd[i] = val;
    } else if j > i && j - i <= skld[j + 1] - skld[j] {
        let k = skld[j] + j - i - 1;
        vkgs[k] = val;
    } else if j < i && i - j <= ikld[i + 1] - ikld[i] {
        let k = ikld[i] + i - j - 1;
        vkgi[k] = val;
    } else {
        //println!("i={} j={}", i, j);
        //println!("Out of profile access, ignored !");
        return Err(());
    }

    Ok(())
}

#[allow(dead_code)]
fn get_csr(val_csr: &Vec<f64>, row_start: &Vec<usize>, jv: &Vec<usize>, i: usize, j: usize) -> f64 {
    let mut val = 0.;
    for iv in row_start[i]..row_start[i + 1] {
        if jv[iv] == j {
            val = val_csr[iv];
        }
    }
    val
}

// full LU tools for debug

#[allow(clippy::needless_range_loop)]
pub fn plu_facto(a: &mut Vec<Vec<f64>>, sigma: &mut Vec<usize>) {
    // initialize permutation to identity
    for (p, sig) in sigma.iter_mut().enumerate() {
        *sig = p;
    }
    let n = sigma.len();
    assert_eq!(a.len(), n);
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

pub fn doolittle_lu(a: &mut Vec<Vec<f64>>) {
    let n = a.len();
    // pivot loop
    for k in 1..n {
        //update row left to the pivot
        for j in 0..k {
            for p in 0..j {
                a[k][j] -= a[k][p] * a[p][j];
            }
            a[k][j] /= a[j][j];
            //println!("L({},{})={}", k, j, a[k][j]);
        }
        //update column p over the pivot
        for i in 0..k + 1 {
            for p in 0..i {
                a[i][k] -= a[i][p] * a[p][k];
            }
            //println!("U({},{})={}", i, k, a[i][k]);
        }
    }
}

#[allow(clippy::needless_range_loop)]
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

#[allow(clippy::needless_range_loop)]
pub fn gauss_solve(a: &Vec<Vec<f64>>, sigma: &Vec<usize>, x: &mut Vec<f64>) {
    let n = a.len();
    assert_eq!(n, x.len());
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

fn main() {
    let mut coo: Vec<(usize, usize, f64)> = vec![];

    let n = 5;

    for i in 0..n {
        coo.push((i, i, 2.));
    }
    for i in 0..n - 1 {
        coo.push((i, i + 1, -1.));
        coo.push((i + 1, i, -1.));
    }

    coo.push((0, 0, 0.));
    coo.push((n - 1, n - 1, 0.));

    coo.push((3, 0, 0.1));
    coo.push((1, 3, 0.1));

    // coo = vec![];

    // for i in 0..n {
    //     for j in 0..n {
    //         coo.push((i, j, 1. / (i + j + 1) as f64));
    //     }
    // }

    let mut sky = SkyMat::new(coo.clone());

    let u = vec![0.; 5];

    println!("Au={:?}", sky.vec_mult(&u));

    sky.compress();
    println!("Au={:?}", sky.vec_mult(&u));
    //sky.print();
    sky.coo_to_sky();
    println!("sky={:?}", sky);

    println!("LU facto");
    sky.factolu().unwrap();
    sky.print_lu();

    let n = 5;
    let mut a: Vec<Vec<f64>> = vec![vec![0. as f64; n]; n];
    //let mut sigma = vec![0; n];

    coo.iter().for_each(|(i, j, v)| {
        a[*i][*j] += *v;
    });

    doolittle_lu(&mut a);

    let mut erreur = 0.;
    println!("doolittle");
    for i in 0..n {
        for j in 0..n {
            print!("{} ", fmt_f64(a[i][j], FMT));
            erreur += (a[i][j] - sky.get(i, j)).abs();
        }
        println!();
    }
    println!("Facto error={}", erreur);

    let x0 = (1..n + 1).map(|i| i as f64).collect();

    let b = sky.vec_mult(&x0);

    let x = sky.solve(b.clone());

    println!("x0={:?}", x0);
    println!("x={:?}", x);
}
