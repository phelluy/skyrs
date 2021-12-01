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

fn main() {
    let val = vec![2., 2., 2.];

    let iv = vec![0, 1, 2];
    let jv = vec![0, 1, 2];

    print_coo(val, iv, jv);
}
