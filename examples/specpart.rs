/// Solves the Laplace equation on a square with the
/// finite difference method.
/// This example requires a working installation of Python3
/// and Matplotlib: the PATH environment variable has to be
/// correctly set.
/// If Matplotlib is not available, the example will run but
/// the results will not show up.
fn main() {
    // grid definition
    let nx = 20;
    let ny = 40;

    // let lx = 1.;
    // let ly = 1.;
    let lx = nx as f64;
    let ly = ny as f64;

    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    let dex = 1.;
    let dy = 1.;

    println!("Assembling...");
    let mut vecval = vec![];

    let n = (nx + 1) * (ny + 1);
    for k in 0..n {
        let i = k % (nx + 1);
        let j = k / (nx + 1);
        vecval.push((k, k, 4.));
    }

    for i in 0..nx {
        for j in 0..ny + 1 {
            let k1 = j * (nx + 1) + i;
            let k2 = j * (nx + 1) + i + 1;
            vecval.push((k1, k2, -1.));
            vecval.push((k2, k1, -1.));
        }
    }

    for j in 0..ny {
        for i in 0..nx + 1 {
            let k1 = j * (nx + 1) + i;
            let k2 = (j + 1) * (nx + 1) + i;
            vecval.push((k1, k2, -1.));
            vecval.push((k2, k1, -1.));
        }
    }

    // finding the largest eigenvalue/eigenvector
    println!("Solving...");
    let m = skyrs::Sky::new(vecval);

    let mut u: Vec<f64> = (0..n).map(|i| i as f64).collect();

    for iter in 0..10 {
        let mut v = m.vec_mult(&u);
        let moyv = v.iter().sum() / n as f64;
        v.iter_mut().for_each(|u| *u -= moyv);
        let normv: f64 = v.iter().map(|u| u * u).sum::<f64>().sqrt();
        println!("normu={}", normv);
        v.iter_mut().for_each(|u| (*u /= normv);
        u = v;
        }

    //let  = m.solve(vec![1.; n]).unwrap();

    // plot
    let xp: Vec<f64> = (0..nx + 1).map(|i| i as f64 * dx).collect();
    let yp: Vec<f64> = (0..ny + 1).map(|i| i as f64 * dy).collect();

    println!("OK");

    println!("Trying to plot...");
    plotpy(xp, yp, u);
}

#[allow(dead_code)]
fn displaymat(m: &skyrs::Sky) {
    println!("Matrix=");
    m.print_coo();
    println!("L-I+U");
    m.print_lu();
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
