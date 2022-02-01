use rayon::prelude::*;
use std::collections::HashMap;

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
/// let b = sky.dot(&x0);
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
    /// row start array in coo (when compressed)
    rowstart: Vec<usize>,
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
    /// permutation
    sigma: Vec<usize>,
    inv_sigma: Vec<usize>,
    /// subdomain indicator
    color: Vec<f64>,
    bisection: HashMap<(usize, usize), (usize, usize, usize, usize)>,
}

const WIDTH: usize = 12;
const PREC: usize = 5;
const EXP: usize = 2;

const FMT: (usize, usize, usize) = (WIDTH, PREC, EXP);

/// Renumbers of graph given in coo format
/// Performs a BFS with a numbering of nodes
/// with less neighbours first
#[allow(dead_code)]
fn coo_renum_bfs(mut coo: Vec<(usize, usize)>) -> Vec<usize> {
    // for each node, count its neighbors
    let mut n = 0;
    for (i, j) in coo.iter() {
        n = n.max((*i).max(*j));
    }
    n += 1;
    let mut neighb = vec![0; n];
    for (i, _j) in coo.iter() {
        neighb[*i] += 1;
    }

    //sort the adjacency list by node index and number of neighbours
    coo.par_sort_unstable_by(|(i1, j1), (i2, j2)| {
        (i1, neighb[*j1], j1).cmp(&(i2, neighb[*j2], j2))
    });

    let mut rowstart: Vec<usize> = vec![];
    rowstart.push(0);
    let mut count = 0;

    rowstart.push(0);
    let (mut iprev, _) = coo[0];
    coo.iter().for_each(|(i, _j)| {
        if iprev != *i {
            rowstart.push(count);
            iprev = *i;
        }
        count += 1;
    });
    rowstart.push(count);

    let mut permut: Vec<usize> = vec![];
    let mut visited: Vec<bool> = vec![false; n];
    // the starting node is visited
    let start = 0;
    visited[start] = true;
    permut.push(start);

    // for (k,s) in split.iter().enumerate() {
    //     if *s == 2 {
    //         permut.push(k);
    //         visited[k] = true;
    //     }
    // }

    //now performs the BFS
    for loc in 0..n {
        // if nodes are exhausted take the first one which is not
        // visited. This may happen because the sub-graphs
        // are not necessarily connected...
        if permut.len() <= loc {
            let rs = visited.iter().position(|visited| !visited);
            let rs = rs.unwrap();
            permut.push(rs);
            visited[rs] = true;
        };
        let sloc = permut[loc];
        // visit the nodes touching physical node sloc
        for i in rowstart[sloc]..rowstart[sloc + 1] {
            let (_, j) = coo[i]; // j: physical index
            if !visited[j] {
                visited[j] = true;
                permut.push(j);
            }
        }
    }
    permut
}

/// Constant size formatter: useful for debug.
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

/// Optimized scalar products of a sub-row of L
/// with a sub-column of U (used in the L triangulation).
/// Uses a BLAS library (which has to be installed on the system).
/// Functional version
#[inline(always)] // probably useless, but...
fn scall(
    i: usize,
    j: usize,
    kmin: usize, // décalage dans utab et ltab
    prof: &[usize],
    ltab: &[f64],
    sky: &[usize],
    utab: &[Vec<f64>],
) -> f64 {
    let pmin = prof[i].max(sky[j]);
    let (lmin, lmax) = (pmin - prof[i], j - prof[i]);
    let (umin, _umax) = (pmin - sky[j], j - sky[j]);
    let size = (lmax - lmin) as i32;
    //let pl = &(ltab[i - kmin][lmin]);
    let pl = &(ltab[lmin]);
    //    println!("j={} kmin={}",j,kmin);
    let pu = &(utab[j - kmin][umin]);
    let scal = if size > 0 {
        unsafe { sysblas::cblas_ddot(size, pl, 1, pu, 1) }
    } else {
        0.
    };
    scal
}

/// Optimized scalar products of a sub-row of L.
/// with a sub-column of U (used in the U triangulation).
/// Uses a BLAS library (which has to be installed on the system).
#[inline(always)] // probably useless, but...
fn scalu(
    i: usize,
    j: usize,
    kmin: usize, // décalage dans utab et ltab
    prof: &[usize],
    ltab: &[Vec<f64>],
    sky: &[usize],
    utab: &[f64],
) -> f64 {
    let pmin = prof[i].max(sky[j]);
    let (lmin, lmax) = (pmin - prof[i], i - prof[i]);
    let (umin, _umax) = (pmin - sky[j], i - sky[j]);
    let size = (lmax - lmin) as i32;
    let scal = if size > 0 {
        let pl = &(ltab[i - kmin][lmin]);
        let pu = &(utab[umin]);
        unsafe { sysblas::cblas_ddot(size, pu, 1, pl, 1) }
    } else {
        0.
    };
    scal
}

/// Performs an LU decomposition on a range of
/// rows/columns of a sparse matrix structure
/// with the Doolittle algorithm. Functional
/// version without test on vanishing pivot.
pub fn factolu_par(
    kmin: usize,
    kmax: usize,
    kmem: usize, // can be kmin or zero
    prof: &[usize],
    ltab: &mut [Vec<f64>],
    sky: &[usize],
    utab: &mut [Vec<f64>],
) {
    for k in kmin..kmax {
        for j in prof[k]..k {
            let mut lkj = ltab[k - kmem][j - prof[k]];
            lkj -= scall(k, j, kmem, prof, ltab[k - kmem].as_slice(), sky, utab);
            lkj /= utab[j - kmem][j - sky[j]];
            ltab[k - kmem][j - prof[k]] = lkj;
        }
        for i in sky[k].max(1)..k + 1 {
            let mut uik = utab[k - kmem][i - sky[k]];
            uik -= scalu(i, k, kmem, prof, ltab, sky, utab[k - kmem].as_slice());
            utab[k - kmem][i - sky[k]] = uik;
        }
    }
}

/// Performs an LU decomposition on a range of
/// rows/columns of a sparse matrix structure
/// with the Doolittle algorithm.
/// Specialized version for the last pass on the
/// bisected matrix.
pub fn factolu_par2(
    kmin: usize,
    kmax: usize,
    kmem: usize, // can be kmin or zero
    prof: &[usize],
    ltab: &mut [Vec<f64>],
    sky: &[usize],
    utab: &mut [Vec<f64>],
) {
    // external part (split into two k loops)
    ltab.par_iter_mut()
        .skip(kmin - kmem)
        .take(kmax - kmin)
        .enumerate()
        .for_each(|(kr, lt)| {
            let k = kr + kmin;
            for j in prof[k]..kmin {
                let mut lkj = lt[j - prof[k]];
                lkj -= scall(k, j, kmem, prof, lt.as_slice(), sky, utab);
                lkj /= utab[j - kmem][j - sky[j]];
                lt[j - prof[k]] = lkj;
            }
        });
    utab.par_iter_mut()
        .skip(kmin - kmem)
        .take(kmax - kmin)
        .enumerate()
        .for_each(|(kr, ut)| {
            let k = kr + kmin;
            for i in sky[k]..kmin {
                let mut uik = ut[i - sky[k]];
                uik -= scalu(i, k, kmem, prof, ltab, sky, ut.as_slice());
                ut[i - sky[k]] = uik;
            }
        });
    // last block
    for k in kmin..kmax {
        for j in prof[k].max(kmin)..k {
            let mut lkj = ltab[k - kmem][j - prof[k]];
            lkj -= scall(k, j, kmem, prof, ltab[k - kmem].as_slice(), sky, utab);
            lkj /= utab[j - kmem][j - sky[j]];
            ltab[k - kmem][j - prof[k]] = lkj;
        }
        for i in sky[k].max(kmin)..k + 1 {
            let mut uik = utab[k - kmem][i - sky[k]];
            uik -= scalu(i, k, kmem, prof, ltab, sky, utab[k - kmem].as_slice());
            utab[k - kmem][i - sky[k]] = uik;
        }
    }
}

/// Recursion of the above algorithms.
/// Remark: if the bisection table is void
/// this will simply run the sequential algorithm.
pub fn factolu_recurse(
    kmin: usize,
    kmax: usize,
    prof: &[usize],
    ltab: &mut [Vec<f64>],
    sky: &[usize],
    utab: &mut [Vec<f64>],
    bisection: &HashMap<(usize, usize), (usize, usize, usize, usize)>,
) {
    let bis = bisection.get(&(kmin, kmax));
    match bis {
        None => {
            // finest bisection level or no bisection at all:
            factolu_par(kmin, kmax, kmin, prof, ltab, sky, utab);
        }
        Some(&(n0, n1, n2, n3)) => {
            let (mut ltab0, mut ltab1) = ltab.split_at_mut(n1 - kmin);
            let (mut utab0, mut utab1) = utab.split_at_mut(n1 - kmin);
            rayon::join(
                || {
                    factolu_recurse(n0, n1, prof, &mut ltab0, sky, &mut utab0, bisection);
                },
                || {
                    factolu_recurse(n1, n2, prof, &mut ltab1, sky, &mut utab1, bisection);
                },
            );
            let imin = n2;
            let imax = n3;
            let imem = n0;
            // Specialized LU algo for the third block
            factolu_par2(imin, imax, imem, prof, ltab, sky, utab);
        }
    }
}

impl Sky {
    /// Constructs a matrix from a coordinate (coo) array.
    /// The coo array is compressed during the process:
    /// the values with same indices are added together.
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
        let sigma: Vec<usize> = (0..nrows).collect();
        let inv_sigma = sigma.clone();
        let color: Vec<f64> = vec![0.; nrows];
        let bisection = HashMap::new();
        let mut sky = Sky {
            coo,
            rowstart: vec![],
            nrows,
            ncols,
            sky: vec![],
            prof: vec![],
            ltab: vec![vec![]; nrows],
            utab: vec![vec![]; ncols],
            sigma,
            inv_sigma,
            color,
            bisection,
        };
        sky.compress();
        sky
    }
    /// Reorders the nodes in the range nmin..nmax:
    /// first apply a BFS on this range,
    /// then splits the nodes into two and
    /// puts the middle nodes at the end.
    /// Returns the boundaries of the three collections of nodes.
    pub fn bisection_bfs(&mut self, nmin: usize, nmax: usize) -> (usize, usize, usize, usize) {
        let n = self.nrows;
        assert_eq!(n, self.ncols);
        // size of the local permutation in sigma[nmin..nmax]
        let ns = nmax - nmin;
        let mut permut: Vec<usize> = vec![];
        // remember the locally visited nodes
        let mut visited: Vec<bool> = vec![false; ns];
        // array for finding the boundary nodes
        let mut cross: Vec<bool> = vec![false; ns];
        // initial node: this a logical node in nmin..nmax
        let start = nmin - nmin;
        // the starting node is visited
        visited[start] = true;
        permut.push(start + nmin);

        // middle point for cutting the list
        let mid = nmin + (nmax - nmin) / 2;

        // for loc in nmin..nmax {
        //     let js = self.inv_sigma[loc];
        //     permut.push(js);
        // }

        // now start a bfs on a possibly non-connected
        // sub-graph
        for loc in nmin..nmax {
            // if nodes are exhausted take the first one which is not
            // visited. This may happen because the sub-graphs
            // are not necessarily connected...
            if permut.len() <= loc - nmin {
                let rs = visited.iter().position(|visited| !visited);
                let rs = rs.unwrap() + nmin;
                permut.push(rs);
                visited[rs - nmin] = true;
            };
            let sloc = self.sigma[permut[loc - nmin]];
            // visit the nodes touching physical node sloc
            for i in self.rowstart[sloc]..self.rowstart[sloc + 1] {
                let (_, j, _) = self.coo[i]; // j: physical index
                let js = self.inv_sigma[j]; // js: logical index in nmin..nmax
                if js < ns + nmin && !visited[js - nmin] {
                    visited[js - nmin] = true;
                    permut.push(js);
                    let jindex = nmin + permut.len() - 1;
                    // mark boundary nodes
                    // <---- THERE IS A DEFAULT HERE ------------>
                    // we assume wrongly that the non zero terms
                    // are disposed symmetrically...
                    // works only if coo_sym has been called before
                    if loc < mid && jindex >= mid {
                        cross[js - nmin] = true;
                    }
                }
            }
        }
        // reverse the end of the list and compute the
        // sub-list boundaries
        permut[mid - nmin..nmax - nmin].reverse();
        let n1 = permut.iter().map(|i| cross[i - nmin]).position(|v| v);
        let n1 = match n1 {
            Some(k) => k + nmin,
            None => nmax,
        };

        // update the global permutation
        for i in nmin..nmax {
            permut[i - nmin] = self.sigma[permut[i - nmin]];
        }

        self.sigma[nmin..nmax].clone_from_slice(&permut[0..(nmax - nmin)]);

        //update inverse permutation
        for i in nmin..nmax {
            self.inv_sigma[self.sigma[i]] = i;
        }
        let n0 = mid;
        let n2 = nmax;
        (nmin, n0, n1, n2)
    }

    /// Reorders the nodes in the range nmin..nmax:
    /// first splits the nodes into two
    /// thanks to metis partitioner.
    /// Detect the interface nodes and put them at the end.
    /// Returns the boundaries of the three collections of nodes.
    pub fn bisection_metis(&mut self, nmin: usize, nmax: usize) -> (usize, usize, usize, usize) {
        // split
        let mid = nmin + (nmax - nmin) / 2;
        let mut split: Vec<usize> = (0..nmax - nmin)
            .map(|is| if is + nmin < mid { 0 } else { 1 })
            .collect();
        // detect boundary nodes
        for is in nmin..nmax {
            let i = self.sigma[is];
            for vois in self.rowstart[i]..self.rowstart[i + 1] {
                let (_, j, _) = self.coo[vois];
                let js = self.inv_sigma[j];
                //println!("permut={:?} js={}", self.sigma, js);
                if js < nmax && js >= nmin {
                    if split[is - nmin] == 0 && split[js - nmin] != 0 {
                        split[js - nmin] = 2;
                    }
                }
            }
        }
        // construct permutation
        let mut permut: Vec<usize> = vec![];
        let mut n1 = 0;
        split.iter().enumerate().for_each(|(is, sp)| {
            if *sp == 0 {
                permut.push(is + nmin);
                n1 += 1;
            }
        });
        let mut n2 = n1;
        split.iter().enumerate().for_each(|(is, sp)| {
            if *sp == 1 {
                permut.push(is + nmin);
                n2 += 1;
            }
        });
        let mut n3 = n2;
        split.iter().enumerate().for_each(|(is, sp)| {
            if *sp == 2 {
                permut.push(is + nmin);
                n3 += 1;
            }
        });
        assert_eq!(n3 + nmin, nmax);

        // update the global permutation
        for i in nmin..nmax {
            permut[i - nmin] = self.sigma[permut[i - nmin]];
        }

        self.sigma[nmin..nmax].clone_from_slice(&permut[0..(nmax - nmin)]);

        //update inverse permutation
        for i in nmin..nmax {
            self.inv_sigma[self.sigma[i]] = i;
        }
        (nmin, n1 + nmin, n2 + nmin, nmax)
    }

    /// Recurses the above algorithm until each matrix is small enough.
    pub fn bisection_iter(&mut self, nmin: usize, nmax: usize) {
        let n = self.nrows;
        // estimate of the final domains
        let ncpus = 1; // more seems to be slower :-(
        if nmax - nmin > (n / ncpus) {
            let (nb, n0, n1, n2) = self.bisection_bfs(nmin, nmax);
            self.bisection.insert((nmin, nmax), (nb, n0, n1, n2));
            self.color[nmin..n0]
                .iter_mut()
                .for_each(|c| *c = nmin as f64);
            self.color[n0..n1].iter_mut().for_each(|c| *c = n0 as f64);
            self.color[n1..n2].iter_mut().for_each(|c| *c = n1 as f64);
            self.bisection_iter(nmin, n0);
            self.bisection_iter(n0, n1);
        }
    }

    pub fn print_bisection(&self) {
        println!("bisection={:?}", self.bisection);
    }

    /// Renumbers the nodes with a Breadth First Search (BFS).
    /// The matrix graph is supposed to be connected and symmetrized
    pub fn bfs_renumber(&mut self, start: usize) {
        let n = self.nrows;
        assert_eq!(n, self.ncols);
        let mut permut: Vec<usize> = vec![];
        // remember the locally visited nodes
        let mut visited: Vec<bool> = vec![false; n];
        visited[start] = true;
        permut.push(start);

        for loc in 0..n {
            let sloc = permut[loc];
            for i in self.rowstart[sloc]..self.rowstart[sloc + 1] {
                let (_, j, _) = self.coo[i];
                if !visited[j] {
                    visited[j] = true;
                    permut.push(j);
                }
            }
        }
        assert!(permut.len() == n, "The graph matrix is not connected.");
        permut[0..n].reverse();
        self.set_permut(permut);
    }
    // pub fn bfs_renumber(&mut self, start: usize) {
    //     let n = self.nrows;
    //     assert_eq!(n, self.ncols);

    //     let coos: Vec<(usize,usize)> = self.coo.iter().map(|(i,j,_v)| (*i,*j)).collect();

    //     let mut permut = coo_renum_bfs(coos.clone());

    //     assert!(permut.len() == n, "The graph matrix is not connected.");
    //     permut[0..n].reverse();
    //     self.set_permut(permut);
    // }

    /// Defines a new nodes permutation.
    /// This invalidates the LU decomposition.
    pub fn set_permut(&mut self, permut: Vec<usize>) {
        self.sigma = permut;
        // checks
        assert_eq!(self.nrows, self.ncols);
        let n = self.nrows;
        let mut inv_permut: Vec<usize> = vec![n; n];
        for i in 0..n {
            inv_permut[self.sigma[i]] = i;
        }
        for i in 0..n {
            assert_eq!(self.sigma[inv_permut[i]], i);
        }
        self.inv_sigma = inv_permut;

        // invalidate a possible LU decomposition
        self.invalidate();
    }

    /// Invalidates a possible LU decomposition.
    fn invalidate(&mut self) {
        let n = self.nrows;
        // self.sigma = (0..self.nrows).collect();
        // self.inv_sigma = self.sigma.clone();
        self.color = vec![0.; n];
        self.bisection = HashMap::new();
        self.sky = vec![];
        self.prof = vec![];
        self.ltab = vec![vec![]; n];
        self.utab = vec![vec![]; n];
    }

    /// Full print of the coo matrix.
    /// Don't use this on big matrices !
    #[allow(dead_code)]
    pub fn print_coo(&self) {
        // first search the size of the matrix
        if self.coo.is_empty() {
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
            println!();
        }
    }

    /// Plots the sparsity pattern of the matrix on
    /// a picture with np x np pixels.
    /// Each pixel represents a non-zeros count.
    pub fn plot(&self, np: usize) {
        let n = self.ncols;
        assert_eq!(n, self.nrows);
        let npmax = np * np;
        let xp: Vec<f64> = (0..np).map(|i| i as f64 * n as f64 / np as f64).collect();
        let yp: Vec<f64> = (0..np)
            .map(|i| (np - i) as f64 * n as f64 / np as f64)
            .collect();
        let mut zp = vec![0.; npmax];
        for ks in 0..n {
            for js in self.prof[ks]..ks {
                let ip = (ks * np) / n;
                let jp = (js * np) / n;
                zp[ip * np + jp] += 1.;
            }
            for is in self.sky[ks]..ks {
                let ip = (is * np) / n;
                let jp = (ks * np) / n;
                zp[ip * np + jp] += 1.;
            }
        }
        plotpy(xp, yp, zp);
    }

    /// Gets an array of colored nodes.
    /// Used for debug.
    pub fn get_sigma(&self) -> Vec<f64> {
        let n = self.nrows;
        let mut c = vec![0.; n];
        self.color
            .iter()
            .enumerate()
            .for_each(|(k, col)| c[self.sigma[k]] = *col);
        c
    }

    /// Gets the inverse permutation.
    /// Used for debug.
    pub fn get_inv_sigma(&self) -> Vec<usize> {
        self.inv_sigma.clone()
    }

    /// Fully prints the LU decomposition.
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

    /// Returns the value at position (i,j) in L-I+U.
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

    /// Returns the value at position (i,j) in L-I+U.
    /// Fails if (i,j) is not in the profile or the skyline.
    /// Used for debug.
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

    /// Sets the value (i,j) in L-I+U.
    /// Fails if (i,j) is not in the skyline or in the profile.
    fn set_lu(&mut self, i: usize, j: usize, v: f64) {
        if i > j {
            self.set_l(i, j, v);
        } else {
            self.set_u(i, j, v);
        }
    }

    /// Sets the value (i,j) in L-I+U.
    /// Does nothing if (i,j) is not in the skyline or in the profile.
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

    /// Gets the (i,j) value in L.
    /// Fails if (i,j) is not in the profile
    /// or if i == j
    #[inline(always)] // probably useless, but...
    fn get_l(&self, i: usize, j: usize) -> f64 {
        self.ltab[i][j - self.prof[i]]
    }

    /// Get the (i,j) value in U.
    /// Fails if (i,j) is not in the skyline.
    #[inline(always)] // probably useless, but...
    fn get_u(&self, i: usize, j: usize) -> f64 {
        self.utab[j][i - self.sky[j]]
    }

    /// Sets the (i,j) value in L.
    /// Fails if (i,j) is not in the profile
    /// or if i == j
    #[inline(always)] // probably useless, but...
    fn set_l(&mut self, i: usize, j: usize, val: f64) {
        self.ltab[i][j - self.prof[i]] = val;
    }

    /// Sets the (i,j) value in U.
    /// Fails if (i,j) is not in the skyline.
    #[inline(always)] // probably useless, but...
    fn set_u(&mut self, i: usize, j: usize, val: f64) {
        self.utab[j][i - self.sky[j]] = val;
    }

    /// Add fake zeros for symmetrizing the structure of the matrix.
    fn coo_sym(&mut self) {
        let nz0 = self.coo.len();
        let mut _count = 0;
        let mut ncols = 0;
        let mut nrows = 0;
        for k in 0..nz0 {
            let (i, j, _v) = self.coo[k];
            ncols = ncols.max(j);
            nrows = ncols.max(i);
            // search a non-zero value of the form (j,i,v)
            let jstart = self.rowstart[j];
            let jend = self.rowstart[j + 1];
            let tr = self.coo[jstart..jend]
                .iter()
                .position(|(j0, i0, _v0)| *i0 == i && *j0 == j);
            match tr {
                None => {
                    self.coo.push((j, i, 0.));
                    _count += 1;
                }
                Some(_) => {}
            }
        }
        //println!("Add {} elem for sym", _count);
        // then add a few extradiagonal terms
        // for ensuring a connected graph
        assert_eq!(nrows, ncols, "The matrix must be a square matrix");
        // let n = nrows + 1;
        // for i in 0..n-1  {
        //     self.coo.push((i, i + 1, 0.));
        //     self.coo.push((i + 1, i, 0.));
        // }
        self.compress();
    }

    /// Sorts the coo array and combines values with the same (i,j) indices.
    pub fn compress(&mut self) {
        if self.coo.is_empty() {
            return;
        };

        // lexicographic sorting for ensuring that identical entries
        // are near to each other
        self.coo
            .par_sort_unstable_by(|(i1, j1, _v1), (i2, j2, _v2)| (i1, j1).cmp(&(i2, j2)));

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
        let mut rowstart: Vec<usize> = vec![];
        rowstart.push(0);
        let mut count = 0;
        let mut is = 0;

        self.coo.iter().enumerate().for_each(|(k, (i, _j, _v))| {
            if *i != is {
                rowstart.push(k);
                is += 1;
                count += 1;
            }
        });
        rowstart.push(self.coo.len());
        self.rowstart = rowstart;
        // println!("{:?}",rowstart);
        // println!("{:?}",self.coo);
        // panic!();
    }

    /// Matrix vector product using the coo array.
    /// Sequential version.
    pub fn vec_mult_slow(&self, u: &Vec<f64>) -> Vec<f64> {
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

    /// Matrix vector product using the coo array.
    /// Parallel version.
    pub fn vec_mult(&self, u: &Vec<f64>) -> Vec<f64> {
        let mut v: Vec<f64> = vec![0.; self.nrows];
        if u.len() != self.ncols {
            panic!(
                "ncols={} is not equal to vector length={}",
                self.ncols,
                u.len()
            );
        };

        v.par_iter_mut().enumerate().for_each(|(i, v)| {
            self.coo[self.rowstart[i]..self.rowstart[i + 1]]
                .iter()
                .for_each(|coo| {
                    *v += coo.2 * u[coo.1];
                });
        });
        v
    }

    /// Matrix vector product using the coo array.
    /// Parallel version. Alias of the previous function.
    pub fn dot(&self, u: &Vec<f64>) -> Vec<f64> {
        self.vec_mult(u)
    }

    /// Converts the coo array to the skyline format internally.
    pub fn coo_to_sky(&mut self) {
        assert_eq!(self.nrows, self.ncols);
        let n = self.nrows;
        if n == 0 {
            return;
        }
        let mut prof: Vec<usize> = (0..n).collect();
        let mut sky: Vec<usize> = (0..n).collect();
        self.coo.iter().for_each(|(i, j, _v)| {
            let ip = self.inv_sigma[*i];
            let jp = self.inv_sigma[*j];
            if jp > ip {
                sky[jp] = sky[jp].min(ip);
            } else {
                prof[ip] = prof[ip].min(jp);
            }
        });
        // non symmetric profile
        self.prof = prof;
        self.sky = sky;

        // symmetric profile
        // self.prof = prof.iter().zip(sky.iter()).map(|(p,s)| (*p).min(*s)).collect();
        // self.sky = self.prof.clone();

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

        for k in 0..self.coo.len() {
            let (i, j, v) = self.coo[k];
            let (ip, jp) = (self.inv_sigma[i], self.inv_sigma[j]);
            self.set_lu(ip, jp, v);
        }
    }

    /// Returns the number of non-zero stored values.
    pub fn get_nnz(&self) -> usize {
        let mut nnz = 0;
        let n = self.nrows;
        for i in 0..n {
            nnz += self.utab[i].len() + self.ltab[i].len();
        }
        nnz
    }

    /// Performs an LU decomposition on the sparse matrix
    /// with the Doolittle algorithm.
    /// Version with a for loop for debug.
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
    /// with a sub-column of U (used in the L triangulation).
    /// Uses a BLAS library (which has to be installed on the system).
    #[inline(always)] // probably useless, but...
    fn scall(&self, i: usize, j: usize) -> f64 {
        let pmin = self.prof[i].max(self.sky[j]);
        let (lmin, lmax) = (pmin - self.prof[i], j - self.prof[i]);
        let (umin, umax) = (pmin - self.sky[j], j - self.sky[j]);
        let noblas = false;
        let scal = if noblas {
            let iiter = self.ltab[i][lmin..lmax].iter();
            let uiter = self.utab[j][umin..umax].iter();
            //slow method
            let locscal = iiter.zip(uiter).map(|(&l, &u)| l * u).sum();
            locscal
        } else {
            // method with blas/lapack
            let size = (lmax - lmin) as i32;
            let pl = &(self.ltab[i][lmin]);
            let pu = &(self.utab[j][umin]);
            if size > 0 {
                unsafe { sysblas::cblas_ddot(size, pl, 1, pu, 1) }
            } else {
                0.
            }
        };
        scal
    }

    /// Optimized scalar products of a sub-row of L.
    /// with a sub-column of U (used in the U triangulation).
    /// Uses a BLAS library (which has to be installed on the system).
    #[inline(always)] // probably useless, but...
    fn scalu(&self, i: usize, j: usize) -> f64 {
        let pmin = self.prof[i].max(self.sky[j]);
        let (lmin, lmax) = (pmin - self.prof[i], i - self.prof[i]);
        let (umin, umax) = (pmin - self.sky[j], i - self.sky[j]);
        let noblas = false;
        let scal = if noblas {
            // slow method
            let iiter = self.ltab[i][lmin..lmax].iter();
            let uiter = self.utab[j][umin..umax].iter();
            let locscal = iiter.zip(uiter).map(|(&l, &u)| l * u).sum();
            locscal
        } else {
            // method with blas/lapack
            let size = (lmax - lmin) as i32;
            if size > 0 {
                let pl = &(self.ltab[i][lmin]);
                let pu = &(self.utab[j][umin]);
                unsafe { sysblas::cblas_ddot(size, pu, 1, pl, 1) }
            } else {
                0.
            }
        };
        scal
    }

    /// Performs an LU decomposition on the sparse matrix structure
    /// with the Doolittle algorithm. Parallel version without
    /// test on vanishing pivot. Faster if the matrix has been
    /// first renumbered by the bisection algorithm.
    /// Runs sequentially otherwise.
    pub fn factolu_par(&mut self) {
        let n = self.nrows;

        // the borrow rules of rust imposes this...
        let prof = self.prof.clone();
        let sky = self.sky.clone();

        // then call the recursive function
        let kmin = 0;
        let kmax = n;
        factolu_recurse(
            kmin,
            kmax,
            prof.as_slice(),
            self.ltab.as_mut_slice(),
            sky.as_slice(),
            self.utab.as_mut_slice(),
            &self.bisection,
        );
    }

    /// Performs an LU decomposition on the sparse matrix
    /// with the Doolittle algorithm.
    /// This version can be read by a human. The optimized
    /// algorithm is in factolu_par.
    pub fn factolu(&mut self) -> Result<(), String> {
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
    /// Triangular solves.
    /// Calls the LU decomposition before if this is not yet done.
    pub fn solve(&mut self, mut bp: Vec<f64>) -> Result<Vec<f64>, String> {
        let m = self.prof.len();
        if m == 0 {
            // necessary for a correct bfs search
            self.coo_sym();
            //self.bisection_iter(0, self.nrows);
            self.bfs_renumber(0);
            self.coo_to_sky();
            self.factolu_par();
            //println!("coo len={}",self.coo.len());
        }
        let m = self.prof.len();
        let mut b: Vec<f64> = (0..m).map(|i| bp[self.sigma[i]]).collect();
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
                b[i] -= self.get_u(i, j + 1) * b[j + 1];
            }
            b[j] /= self.get_u(j, j);
        }
        bp = (0..m).map(|i| b[self.inv_sigma[i]]).collect();
        Ok(bp)
    }

    /// Triangular solves.
    /// Calls the LU decomposition before if this is not yet done.
    /// No renumbering of the unknowns is performed. This allows to keep
    /// the numbering decided by the user
    pub fn solve_norenum(&mut self, mut b: Vec<f64>) -> Result<Vec<f64>, String> {
        let m = self.prof.len();
        if m == 0 {
            self.coo_to_sky();
            self.factolu_par();
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
                b[i] -= self.get_u(i, j + 1) * b[j + 1];
            }
            b[j] /= self.get_u(j, j);
        }
        Ok(b)
    }

    /// Performs a LU decomposition on the full matrix
    /// with the Doolittle algorithm.
    /// Not efficient: used only for debug.
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

    /// Triangular solves with the full structure.
    /// Only here for debug.
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

    /// Performs the LU decomposition with the Gauss method
    /// on the full matrix.
    /// Not efficient: only for debug purpose.
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

/// Inplace Doolittle LU decomposition on a full matrix.
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

/// Plots a 2D data set using matplotlib.
fn plotpy(xp: Vec<f64>, yp: Vec<f64>, zp: Vec<f64>) {
    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;
    {
        let meshfile = File::create("plotpy.dat").unwrap();
        let mut meshfile = BufWriter::new(meshfile); // create a buffer for faster writes...
        xp.iter().for_each(|x| {
            writeln!(meshfile, "{}", x).unwrap();
        });
        writeln!(meshfile).unwrap();
        yp.iter().for_each(|y| {
            writeln!(meshfile, "{}", y).unwrap();
        });
        writeln!(meshfile).unwrap();
        zp.iter().for_each(|z| {
            writeln!(meshfile, "{}", z).unwrap();
        });
    } // ensures that the file is closed

    use std::process::Command;
    Command::new("python3")
        .arg("examples/plot.py")
        .status()
        .expect("Plot failed: you need Python3 and Matplotlib in your PATH.");
}

/// Permutation algorithm used by gauss_solve.
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

/// Triangular solves.
/// plu_solve must be called first
/// and this has to be checked by the user before.
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

/// Test for small special matrices.
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

    //warning: BFS does not work because the 
    //graph of the diagonal matrix is not connected
    //let v = sky.solve(u).unwrap();
    // therefore use the solver without renumbering
    let v = sky.solve_norenum(u).unwrap();
    println!("{:?}", v);

    assert_float_eq!(v, (0..n).map(|i| i as f64 / 2.).collect(), abs_all <= 1e-12);
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

    // necessary because this matrix is not symmetric
    sky.coo_sym();
    sky.compress();

    let v2 = sky.vec_mult(&u);
    println!("Au={:?}", v2);

    v1.iter()
        .zip(v2.iter())
        .for_each(|(v1, v2)| assert!((*v1 - *v2).abs() < 1e-14));

    sky.coo_to_sky();

    //sky.print_lu();

    //sky.factolu().unwrap();
    sky.factolu_par();

    println!("sky={:?}", sky);

    let mut a: Vec<Vec<f64>> = vec![vec![0. as f64; n]; n];
    //let mut sigma = vec![0; n];
    let inv_sigma = sky.get_inv_sigma();

    coo.iter().for_each(|(i, j, v)| {
        let ip = inv_sigma[*i];
        let jp = inv_sigma[*j];
        a[ip][jp] += *v;
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

#[test]
fn small_norenum() {
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
    sky.factolu_par();

    // comparison with the full solver
    let mut a: Vec<Vec<f64>> = vec![vec![0. as f64; n]; n];

    coo.iter().for_each(|(i, j, v)| {
        a[*i][*j] += *v;
    });

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

    let x = sky.solve_norenum(b.clone()).unwrap();

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
