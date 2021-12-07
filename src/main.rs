/// Display coo matrix in full
fn print_coo(val: &Vec<f64>, iv: &Vec<usize>, jv: &Vec<usize>) {
    // first search the size of the matrix
    let imax = iv.iter().max().unwrap() + 1;
    println!("nrows={}", imax);
    let jmax = jv.iter().max().unwrap() + 1;
    println!("ncols={}", jmax);

    let mut full = vec![0.; imax * jmax];

    let nz = val.len();

    assert_eq!(nz, iv.len());
    assert_eq!(nz, jv.len());

    for (v, (i, j)) in val.iter().zip(iv.iter().zip(jv.iter())) {
        full[i * imax + j] = *v;
    }

    println!("full=");
    for i in 0..imax {
        for j in 0..jmax {
            print!("{} ", full[i * jmax + j]);
        }
        println!("");
    }
    //println!("full={:?}", full);
}

/// Display coo matrix in full
fn print_csr(val_csr: &Vec<f64>, row_start: &Vec<usize>, jv: &Vec<usize>) {
    // first search the size of the matrix
    let n = row_start.len() - 1;
    println!("nrows={}", n);
    println!("ncols={}", n);

    println!("full=");
    for i in 0..n {
        for j in 0..n {
            print!("{} ", get_csr(val_csr, row_start, jv, i, j));
        }
        println!("");
    }
    //println!("full={:?}", full);
}

/// Display coo matrix in full
fn print_sky(vkgd: &Vec<f64>, vkgs: &Vec<f64>, vkgi: &Vec<f64>, kld: &Vec<usize>) {
    // first search the size of the matrix
    let n = kld.len() - 1;
    println!("nrows={}", n);
    println!("ncols={}", n);

    println!("full=");
    for i in 0..n {
        for j in 0..n {
            print!("{} ", get_sky(i, j, vkgd, vkgs, vkgi, kld));
        }
        println!("");
    }
    //println!("full={:?}", full);
}

/// Convert a coo matrix to compressed sparse row (csr) format
/// the matrix must be first sorted and compressed !
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

// the following line define a triplet structure with
// lexical ordering on (i,j)
#[derive(Debug, Copy, Clone)]
struct Triplet {
    i: usize,
    j: usize,
    v: f64,
}

use std::cmp::Ordering;

impl Ord for Triplet {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.i, self.j).cmp(&(other.i, other.j))
    }
}

impl PartialOrd for Triplet {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Triplet {
    fn eq(&self, other: &Self) -> bool {
        (self.i, self.j) == (other.i, other.j)
    }
}

impl Eq for Triplet {}

fn coo_sky_extend(val: &mut Vec<f64>, iv: &mut Vec<usize>, jv: &mut Vec<usize>) {
    let imax = iv.iter().max().unwrap() + 1;
    println!("nrows={}", imax);
    let jmax = jv.iter().max().unwrap() + 1;
    println!("ncols={}", jmax);

    assert_eq!(imax, jmax);

    let mut prof = vec![0; imax];

    iv.iter().zip(jv.iter()).for_each(|(&i, &j)| {
        if j > i {
            prof[j] = prof[j].max(j - i);
        } else {
            prof[i] = prof[i].max(i - j);
        }
    });

    println!("prof={:?}", prof);

    // add fake zeros below the skyline
    for i in 0..imax {
        for di in 1..prof[i] + 1 {
            //println!("i={} di={}",i,di);
            val.push(0.);
            iv.push(i);
            jv.push(i - di);
            val.push(0.);
            jv.push(i);
            iv.push(i - di);
        }
    }
}

/// Convert a matrix in triplet format into skyline format
fn coo_to_sky(
    val: Vec<f64>,
    iv: Vec<usize>,
    jv: Vec<usize>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let imax = iv.iter().max().unwrap() + 1;
    println!("nrows={}", imax);
    let jmax = jv.iter().max().unwrap() + 1;
    println!("ncols={}", jmax);

    assert_eq!(imax, jmax);

    let n = imax;

    let mut prof = vec![0; imax];

    iv.iter().zip(jv.iter()).for_each(|(&i, &j)| {
        if j > i {
            prof[j] = prof[j].max(j - i);
        } else {
            prof[i] = prof[i].max(i - j);
        }
    });

    println!("prof={:?}", prof);

    let mut kld = vec![0; n + 1];

    for i in 0..n {
        kld[i + 1] = kld[i] + prof[i];
    }

    // stored values in the upper and lower triangles
    let nnz = kld[n];
    println!("kld={:?} \n nnz={}", kld, nnz);

    let mut vkgd = vec![0.; n];
    let mut vkgs = vec![0.; nnz];
    let mut vkgi = vec![0.; nnz];

    val.iter()
        .zip(iv.iter().zip(jv.iter()))
        .for_each(|(v, (i, j))| {
            set_sky(*i, *j, *v, &mut vkgd, &mut vkgs, &mut vkgi, &kld).unwrap();
        });

    (vkgd, vkgs, vkgi, kld)
}

fn get_sky(
    i: usize,
    j: usize,
    vkgd: &Vec<f64>,
    vkgs: &Vec<f64>,
    vkgi: &Vec<f64>,
    kld: &Vec<usize>,
) -> f64 {
    if i == j {
        vkgd[i]
    } else if j > i && j - i <= kld[j + 1] - kld[j] {
        let k = kld[j] + j - i - 1;
        vkgs[k]
    } else if j < i && i - j <= kld[i + 1] - kld[i] {
        let k = kld[i] + i - j - 1;
        vkgi[k]
    } else {
        0.
    }
}

use std::error::Error;

fn set_sky(
    i: usize,
    j: usize,
    val: f64,
    vkgd: &mut Vec<f64>,
    vkgs: &mut Vec<f64>,
    vkgi: &mut Vec<f64>,
    kld: &Vec<usize>,
) -> Result<(), ()> {
    if i == j {
        vkgd[i] = val;
    } else if j > i && j - i <= kld[j + 1] - kld[j] {
        let k = kld[j] + j - i - 1;
        vkgs[k] = val;
    } else if j < i && i - j <= kld[i + 1] - kld[i] {
        let k = kld[i] + i - j - 1;
        vkgi[k] = val;
    } else {
        println!("i={} j={} kld={:?}", i, j, kld);
        return Err(());
    }

    Ok(())
}

fn coo_sort_compress(
    val: Vec<f64>,
    iv: Vec<usize>,
    jv: Vec<usize>,
) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    // initial verifications
    assert_eq!(val.len(), iv.len());
    assert_eq!(val.len(), jv.len());
    if val.len() == 0 {
        return (val, iv, jv);
    }
    // copy the vector in a vector of tuples
    let mut vtuple: Vec<Triplet> = iv
        .iter()
        .zip(jv.iter())
        .zip(val.iter())
        .map(|((i, j), v)| Triplet {
            i: *i,
            j: *j,
            v: *v,
        })
        .collect();
    vtuple.sort();
    //println!("{:?}", vtuple);
    let mut iv: Vec<usize> = vec![];
    let mut jv: Vec<usize> = vec![];
    let mut val: Vec<f64> = vec![];

    let mut tr1 = vtuple[0];
    iv.push(tr1.i);
    jv.push(tr1.j);
    let mut cval = tr1.v;

    vtuple.iter().skip(1).for_each(|tr2| {
        if *tr2 == tr1 {
            cval += tr2.v;
        } else {
            val.push(cval);
            tr1 = *tr2;
            iv.push(tr1.i);
            jv.push(tr1.j);
            cval = tr1.v;
        }
    });
    // last push
    val.push(cval);

    (val, iv, jv)
}

fn get_csr(val_csr: &Vec<f64>, row_start: &Vec<usize>, jv: &Vec<usize>, i: usize, j: usize) -> f64 {
    let mut iv = row_start[i];
    let mut val = 0.;
    for iv in row_start[i]..row_start[i + 1] {
        if jv[iv] == j {
            val = val_csr[iv];
        }
    }
    val
}

fn csr_gauss_elim(val_csr: &mut Vec<f64>, row_start: &Vec<usize>, jv: &Vec<usize>) {
    // step p (pivot in col. p)
    let n = row_start.len() - 1;
    for p in 0..n - 1 {
        let piv = get_csr(val_csr, row_start, jv, p, p);
        for i in p + 1..n {
            let c = get_csr(val_csr, row_start, jv, i, p) / piv;
            for iv in row_start[i]..row_start[i + 1] {
                let j = jv[iv];
                val_csr[iv] -= c * get_csr(val_csr, row_start, jv, p, j);
            }
            // Li = Li - c Lp
            // ou si LU
            // a[i][p]= c // LU case
        }
    }
}

fn sky_factolu(vkgd: &mut Vec<f64>, vkgs: &mut Vec<f64>, vkgi: &mut Vec<f64>, kld: &Vec<usize>) {
    let n = kld.len() - 1;
    for p in 0..n - 1 {
        let piv = vkgd[p];
        // fill the column p in L
        for i in p + 1..n {
            let c = get_sky(i, p, vkgd, vkgs, vkgi, kld) / piv; // c = a[i,p] /  a[p,p]
            set_sky(i, p, c, vkgd, vkgs, vkgi, kld); // L[i,p] = c
        }
        for i in p + 1..n {
            // use the column p of L for elimination
            // Ui = Ui - c Up   U[i,j] = U[i,j] - c U[p,j] for j >= i
            // diagonal term
            for j in i..n {
                let mut u = get_sky(i, j, vkgd, vkgs, vkgi, kld);
                let c = get_sky(i, p, vkgd, vkgs, vkgi, kld);
                u -= c * get_sky(p, j, vkgd, vkgs, vkgi, kld);
                set_sky(i, j, u, vkgd, vkgs, vkgi, kld);
            }
            // upper diagonal term
        }
    }
}

// full LU tools
#[allow(clippy::needless_range_loop)]
pub fn plu_facto(a: &mut [[f64; NN]; NN], sigma: &mut [usize; NN]) {
    // initialize permutation to identity
    for (p, sig) in sigma.iter_mut().enumerate() {
        *sig = p;
    }

    // elimination in column p
    for p in 0..NN - 1 {
        // search max pivot
        let mut sup = 0.;
        let mut p_max = p;
        for i in p..NN {
            let abs_a_ip = (a[i][p]).abs();
            if sup < abs_a_ip {
                sup = abs_a_ip;
                p_max = i;
            }
        }

        // swap two lines
        for j in 0..NN {
            let aux = a[p][j];
            a[p][j] = a[p_max][j];
            a[p_max][j] = aux;
        }

        // store permutation
        sigma.swap(p, p_max);

        for i in p + 1..NN {
            let c = a[i][p] / a[p][p];
            for j in p + 1..NN {
                a[i][j] -= c * a[p][j];
                a[i][p] = c;
            }
        }
    }
}

pub fn doolittle_facto(a: &mut [[f64; NN]; NN], sigma: &mut [usize; NN]) {
    // initialize permutation to identity
    for (p, sig) in sigma.iter_mut().enumerate() {
        *sig = p;
    }

    // pivot loop
    //a[1][0] = -a[1][0]/a[0][0]; 
    for k in 1..NN {
        //update row left to the pivot
        for j in 0..k {
            for p in 0..j {
                a[k][j] -= a[k][p] * a[p][j];
            }
            a[k][j] /= a[j][j];
            println!("L({},{})={}",k,j,a[k][j]);
        }
        //update column p over the pivot
        for i in 0..k+1 {
            for p in 0..i {
                a[i][k] -= a[i][p] * a[p][k];
            }
            println!("U({},{})={}",i,k,a[i][k]);
        }
    }
}

// fn permutate_in_place(x: &mut [[f64; M]; NN], sigma: &[usize; NN]) {
//     for i in 0..NN {
//         let mut to_swap = sigma[i];
//         while to_swap < i {
//             to_swap = sigma[to_swap];
//         }
//         x.swap(i, to_swap);
//     }
// }

// #[allow(clippy::needless_range_loop)]
// pub fn plu_solve(a: &[[f64; NN]; NN], sigma: &[usize; NN], x: &mut [[f64; M]; NN]) {
//     // permutate x
//     permutate_in_place(x, sigma);

//     for i in 1..NN {
//         for j in 0..i {
//             for iw in 0..M {
//                 x[i][iw] -= a[i][j] * x[j][iw];
//             }
//         }
//     }

//     for iw in 0..M {
//         x[NN - 1][iw] = x[NN - 1][iw] / a[NN - 1][NN - 1];
//     }
//     for i in (0..NN - 1).rev() {
//         for j in i + 1..NN {
//             for iw in 0..M {
//                 x[i][iw] -= a[i][j] * x[j][iw];
//             }
//         }
//         for iw in 0..M {
//             x[i][iw] = x[i][iw] / a[i][i];
//         }
//     }
// }

fn permutate_in_place_one_var(x: &mut [f64; NN], sigma: &[usize; NN]) {
    for i in 0..NN {
        let mut to_swap = sigma[i];
        while to_swap < i {
            to_swap = sigma[to_swap];
        }
        x.swap(i, to_swap);
    }
}

const NN: usize = 5;

#[allow(clippy::needless_range_loop)]
pub fn plu_solve_one_var(a: &[[f64; NN]; NN], sigma: &[usize; NN], x: &mut [f64; NN]) {
    permutate_in_place_one_var(x, sigma);

    for i in 1..NN {
        for j in 0..i {
            x[i] -= a[i][j] * x[j];
        }
    }

    x[NN - 1] = x[NN - 1] / a[NN - 1][NN - 1];
    for i in (0..NN - 1).rev() {
        for j in i + 1..NN {
            x[i] -= a[i][j] * x[j];
        }
        x[i] = x[i] / a[i][i];
    }
}

fn main() {
    let mut val = vec![];

    let mut iv = vec![];
    let mut jv = vec![];

    let n = 5;

    for i in 0..n {
        val.push(2.);
        iv.push(i);
        jv.push(i);
    }
    for i in 0..n - 1 {
        val.push(-1.);
        iv.push(i);
        jv.push(i + 1);
        val.push(-1.);
        iv.push(i + 1);
        jv.push(i);
    }

    val.push(-0.1);
    iv.push(0);
    jv.push(4);

    val.push(-0.9);
    iv.push(0);
    jv.push(4);

    //coo_sky_extend(&mut val, &mut iv, &mut jv);
    let (mut val, mut iv, mut jv) = coo_sort_compress(val, iv, jv);
    print_coo(&val, &iv, &jv);

    let mut a = [[0. as f64; NN]; NN];
    let mut sigma = [0; NN];

    val.iter()
        .zip(iv.iter().zip(jv.iter()))
        .for_each(|(&v, (&i, &j))| {
            a[i][j] = v;
        });

    doolittle_facto(&mut a, &mut sigma);


    let (mut vkgd, mut vkgs, mut vkgi, kld) = coo_to_sky(val, iv, jv);

    // println!(
    //     "vkgd={:?}
    // vkgs={:?}
    // vkgi={:?}
    // kld={:?}",
    //     vkgd, vkgs, vkgi, kld
    // );

    sky_factolu(&mut vkgd, &mut vkgs, &mut vkgi, &kld);

    println!("sky");
    print_sky(&vkgd, &vkgs, &vkgi, &kld);

    println!("doolittle");
    for i in 0..NN {
        for j in 0..NN{
            print!("{} ", a[i][j])
        }
        println!();
    }
    println!();


    // let (mut val_csr, row_start, jv) = coo_to_csr(val, iv, jv);
    // println!("val={:?} row_start={:?} jv={:?}", val_csr, row_start, jv);
    // csr_gauss_elim(&mut val_csr, &row_start, &jv);
    // print_csr(&val_csr, &row_start, &jv);
}
