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

/// Convert a coo matrix to compressed sparse row (csr) format
/// the matrix must be first sorted and copressed !
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

fn csr_gauss_elim(val_csr: &mut Vec<f64>, row_start: &mut Vec<usize>, jv: Vec<usize>) {

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

    coo_sky_extend(&mut val, &mut iv, &mut jv);
    let (mut val, mut iv, mut jv) = coo_sort_compress(val, iv, jv);
    print_coo(&val, &iv, &jv);
    let (val_csr, row_start, jv) = coo_to_csr(val, iv, jv);
    println!("val={:?} row_start={:?} jv={:?}", val_csr, row_start, jv);
}
