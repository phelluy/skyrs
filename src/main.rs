/// Display coo matrix in full
fn print_coo(val: Vec<f64>, iv: Vec<usize>, jv: Vec<usize>) {
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
    println!("full={:?}", full);
}

/// Convert a coo matrix to compressed sparse row (csr) format
fn coo_to_csr(
    val_coo: Vec<f64>,
    iv: Vec<usize>,
    jv: Vec<usize>,
    val_csr: &mut Vec<f64>,
    col_index: &mut Vec<usize>,
    row_start: &mut Vec<usize>,
) {
    // some checks
    assert_eq!(val_coo.len(), iv.len());
    assert_eq!(val_coo.len(), jv.len());
    let nrows = *iv.iter().max().unwrap();
    // first count non zero values in each row
    let mut row_count = vec![0, nrows];
    iv.iter().for_each(|iv| row_count[*iv] += 1);
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

fn coo_sort_compress(val: Vec<f64>, iv: Vec<usize>, jv: Vec<usize>) 
-> (Vec<f64>, Vec<usize>, Vec<usize>) {
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
    println!("{:?}", vtuple);
    let mut iv:Vec<usize>=vec![];
    let mut jv: Vec<usize>=vec![];
    let mut val: Vec<f64>=vec![];
    let mut tr1 = vtuple[0];
    iv.push(tr1.i);
    jv.push(tr1.j);
    let mut cval = tr1.v;

    vtuple.iter().skip(1).for_each(|tr2| if *tr2 == tr1 {
        cval += tr2.v;
    } else {
        val.push(cval);
        tr1 = *tr2;
        iv.push(tr1.i);
        jv.push(tr1.j);
        cval = tr1.v;  
    });
    // last push
    val.push(cval);
    // let iv: Vec<usize> = vtuple.iter().map(|trip| trip.i).collect();
    // let jv: Vec<usize> = vtuple.iter().map(|trip| trip.j).collect();
    // let val: Vec<f64> = vtuple.iter().map(|trip| trip.v).collect();
    println!("{:?}",val);
    (val,iv,jv)
}

fn main() {
    let val = vec![2., 2., 1., 1.];

    let iv = vec![2, 1, 0, 0];
    let jv = vec![2, 1, 0, 0];

    let (val,iv,jv) = coo_sort_compress(val, iv, jv);
    print_coo(val, iv, jv);
}
