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

use rayon::prelude::*; 

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
        let sigma: Vec<usize> = (0..nrows).collect();
        let inv_sigma = sigma.clone();
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
        };
        sky.compress();
        sky
    }
    /// reorder the nodes in the range nmin..nmax
    /// first apply a bfs on this range
    /// then split the nodes into two
    /// put the middle nodes at the end
    /// this is performed in the old permutation
    /// there is a bug: this algorithm do not accept
    /// an initial sigma which is not the identity...
    pub fn bisection_bfs(&mut self, nmin: usize, nmax: usize) -> (usize, usize, usize) {
        let n = self.nrows;
        assert_eq!(n, self.ncols);
        // the coo array must be sorted by row and by col
        let ns = nmax - nmin;
        // for storing the permutation in sigma[nmin..nmax]
        let mut permut: Vec<usize> = vec![];
        // remember the locally visited nodes
        let mut visited: Vec<bool> = vec![false; ns];
        // array for finding the boundary nodes
        let mut cross: Vec<bool> = vec![false; ns];
        // initial node: this must be a physical number
        let start = nmin - nmin;
        // the logical node is visited
        visited[start] = true;
        //permut.push(start);
        permut.push(nmin);
        //println!("init jindex={}",jindex);
        // middle index of the visited list
        let mid = nmin + (nmax - nmin) / 2;
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
            for i in self.rowstart[sloc]..self.rowstart[sloc + 1] {
                let (_, j, _) = self.coo[i];
                let js = self.inv_sigma[j];
                if js < ns && !visited[js - nmin] {
                    visited[js - nmin] = true;
                    permut.push(js);
                    let jindex = permut.len() - 1;
                    //println!("jindex={}",jindex);
                    if loc < mid && jindex >= mid - nmin {
                        cross[js - nmin] = true;
                    }
                }
            }
        }
        permut[mid - nmin..nmax - nmin].reverse();
        //println!("nmin={} nmax={} loc permut={:?}",nmin,nmax,permut);
        let n1 = permut.iter().map(|i| cross[i - nmin]).position(|v| v);
        let n1 = match n1 {
            Some(k) => k + nmin,
            None => nmax,
        };
        //println!
        for i in nmin..nmax {
            permut[i - nmin] = self.sigma[permut[i - nmin]];
        }
        for i in nmin..nmax {
            self.sigma[i] = permut[i - nmin];
        }

        //update inverse permutation
        for i in nmin..nmax {
            self.inv_sigma[self.sigma[i]] = i;
        }

        // let mut end:Vec<usize> = self.sigma[nmax..n].iter().map(|s| *s).collect();
        // permut.append(&mut end);
        //println!("permut {:?}", permut);
        //panic!();
        //println!("cross {:?}", cross);
        //let it:Vec<bool> = permut.iter().map(|i| cross[*i]).collect();
        //println!("cross bis {:?}", it);
        //println!("visited {:?}", visited);
        let n0 = mid;
        let n2 = nmax;
        //println!("nmin={} nmax={} n0={} n1={} n2={}",nmin,nmax,n0,n1,n2);
        //panic!();
        //self.set_permut(self.sigma.clone());
        (n0, n1, n2)
    }

    pub fn bfs_renumber(&mut self, start: usize) {
        let n = self.nrows;
        assert_eq!(n, self.ncols);
        let mut permut: Vec<usize> = vec![];
        // remember the locally visited nodes
        let mut visited: Vec<bool> = vec![false; n];
        // array for finding the boundary nodes
        // initial node: this must be a physical number
        // the logical node is visited
        visited[start] = true;
        //permut.push(start);
        permut.push(start);

        for loc in 0..n {
            // the mesh is supposed to be connected
            for i in self.rowstart[loc]..self.rowstart[loc + 1] {
                let (_, j, _) = self.coo[i];
                if !visited[j] {
                    visited[j] = true;
                    permut.push(j);
                }
            }
        }
        permut[0..n].reverse();
        //println!("permut={:?}",permut);
        self.set_permut(permut);
    }

    /// recurse the above algorithm until each matrix is small enough
    pub fn bisection_iter(&mut self, nmin: usize, nmax: usize) {
        if nmax - nmin >= self.nrows / 32 {
            let (n0, n1, _n2) = self.bisection_bfs(nmin, nmax);
            //println!("nmin={} nmax={} tr={:?}",nmin,nmax,(n0,n1,n2));
            self.bisection_iter(nmin, n0);
            self.bisection_iter(n0, n1);
        }
    }

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
        self.sky = vec![];
        self.prof = vec![];
        self.ltab = vec![vec![]; n];
        self.utab = vec![vec![]; n];
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
            println!();
        }
    }

    // plot the non zero pattern of the matrix
    pub fn plot(&self, np: usize) {
        let n = self.ncols;
        assert_eq!(n, self.nrows);
        //let np = n;
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
        // self.coo.iter().for_each(|(i, j, _v)| {
        //     let is = self.inv_sigma[*i];
        //     let js = self.inv_sigma[*j];
        //     let ip = (is * np) / n;
        //     let jp = (js * np) / n;
        //     zp[ip * np + jp] += if is > js && js >= self.prof[is] {
        //         1.
        //     } else if is <= js && is >= self.sky[js] {
        //         1.
        //     } else {
        //         0.
        //     };
        // });

        plotpy(xp, yp, zp);
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

    /// Return one if (i,j) is in the profile or the skyline
    /// and zero otherwise
    fn get_struct(&self, i: usize, j: usize) -> f64 {
        assert!(i < self.nrows);
        assert!(j < self.ncols);
        if i > j && j >= self.prof[i] {
            1.
        } else if i <= j && i >= self.sky[j] {
            1.
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
    pub fn compress(&mut self) {
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

    /// Matrix vector product using the coo array
    /// sequential version
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

    
    /// Matrix vector product using the coo array
    /// parallel version
    pub fn vec_mult(&self, u: &Vec<f64>) -> Vec<f64> {
        let n = self.nrows;
        let mut v: Vec<f64> = vec![0.; self.nrows];
        if u.len() != self.ncols {
            panic!(
                "ncols={} is not equal to vector length={}",
                self.ncols,
                u.len()
            );
        };
        v.par_iter_mut().enumerate().for_each(|(i,v)| {
            self.coo[self.rowstart[i]..self.rowstart[i+1]].iter().for_each(|coo| {
                *v += coo.2 * u[coo.1];
            });
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
        self.coo.iter().for_each(|(i, j, _v)| {
            let ip = self.inv_sigma[*i];
            let jp = self.inv_sigma[*j];
            // println!("{:?}",self.sigma);
            // println!("{:?}",self.inv_sigma);
            //panic!();
            if jp > ip {
                sky[jp] = sky[jp].min(ip);
            } else {
                prof[ip] = prof[ip].min(jp);
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

        for k in 0..self.coo.len() {
            let (i, j, v) = self.coo[k];
            let (ip, jp) = (self.inv_sigma[i], self.inv_sigma[j]);
            self.set_lu(ip, jp, v);
        }
    }

    pub fn get_nnz(&self) -> usize {
        let mut nnz = 0;
        let n = self.nrows;
        for i in 0..n {
            nnz += self.utab[i].len() + self.ltab[i].len();
        }

        nnz
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
        let (umin, _umax) = (pmin - self.sky[j], j - self.sky[j]);
        // let iiter = self.ltab[i][lmin..lmax].iter();
        // let uiter = self.utab[j][umin..umax].iter();
        // slow method
        // let scal: f64 = iiter.zip(uiter).map(|(&l, &u)| l * u).sum();
        // method with blas/lapack
        let size = (lmax - lmin) as i32;
        let pl = &(self.ltab[i][lmin]);
        let pu = &(self.utab[j][umin]);
        let scal = if size > 0 {
            unsafe { sysblas::cblas_ddot(size, pl, 1, pu, 1) }
        } else {
            0.
        };
        scal
    }

    /// Optimized scalar products of a sub-row of L
    /// with a sub-column of U (used in the U triangulation)
    #[inline(always)] // probably useless, but...
    fn scalu(&self, i: usize, j: usize) -> f64 {
        let pmin = self.prof[i].max(self.sky[j]);
        let (lmin, lmax) = (pmin - self.prof[i], i - self.prof[i]);
        let (umin, _umax) = (pmin - self.sky[j], i - self.sky[j]);
        // slow method
        // let iiter = self.ltab[i][lmin..lmax].iter();
        // let uiter = self.utab[j][umin..umax].iter();
        // let scal: f64 = iiter.zip(uiter).map(|(&l, &u)| l * u).sum();
        // method with blas/lapack
        let size = (lmax - lmin) as i32;
        let scal = if size > 0 {
            let pl = &(self.ltab[i][lmin]);
            let pu = &(self.utab[j][umin]);
            unsafe { sysblas::cblas_ddot(size, pu, 1, pl, 1) }
        } else {
            0.
        };
        scal
    }

    /// Performs a LU decomposition on the sparse matrix
    /// with the Doolittle algorithm
    pub fn factolu(&mut self) -> Result<(), String> {
        //self.coo_to_sky();
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
    pub fn solve(&mut self, mut bp: Vec<f64>) -> Result<Vec<f64>, String> {
        let m = self.prof.len();
        if m == 0 {
            self.coo_to_sky();
            self.factolu()?;
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
                //for i in 0..j {
                b[i] -= self.get_u(i, j + 1) * b[j + 1];
            }
            b[j] /= self.get_u(j, j);
        }
        bp = (0..m).map(|i| b[self.inv_sigma[i]]).collect();
        Ok(bp)
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

/// Plot a 2D data set using matplotlib
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
        writeln!(meshfile, "").unwrap();
        yp.iter().for_each(|y| {
            writeln!(meshfile, "{}", y).unwrap();
        });
        writeln!(meshfile, "").unwrap();
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
