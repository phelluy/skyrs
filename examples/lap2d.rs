/// Solves the Laplace equation on a square with the
/// finite difference method.
/// This example requires a working installation of Python3
/// and Matplotlib: the PATH environment variable has to be
/// correctly set.
/// If Matplotlib is not available, the example will run but
/// the results will not show up.
use rayon::prelude::*;

fn main() {
    ////////////////// Dirichlet ////////////////////
    let lx = 2.;
    let ly = 1.;
    let nx = 300;
    let ny = 300;
    println!("Dirichlet assembly...");
    let vecval = dirichlet(lx, ly, nx, ny);

    let dx = lx / nx as f64;
    let dy = dx;
    let n = (nx + 1) * (ny + 1);

    // linear system resolution
    println!("Compress...");
    let mut m = skyrs::Sky::new(vecval);
    println!("Renumbering...");

    println!("Solving...");
    let f0 = vec![1.; n];

    use std::time::Instant;

    let start = Instant::now();
    let zp = m.solve(f0.clone()).unwrap();
    let duration = start.elapsed();

    println!("Solving time: {:?}", duration);
    println!("nnz={}", m.get_nnz());

    let f = m.vec_mult(&zp);

    let err: f64 = f.iter().zip(f0.iter()).map(|(f, f0)| (f - f0).abs()).sum();
    println!("err={:e}", err / n as f64);

    let xp: Vec<f64> = (0..nx + 1).map(|i| i as f64 * dx).collect();
    let yp: Vec<f64> = (0..ny + 1).map(|i| i as f64 * dy).collect();
    plotpy(xp.clone(), yp.clone(), zp);

    plotpy(xp, yp, m.get_sigma());

    println!("Matrix plot...");
    m.plot(100);

    /////////////// Neumann //////////////////////////////

    println!("Neumann assembly...");
    let vecval = lapgraph(nx, ny);

    let m = skyrs::Sky::new(vecval);

    // benchmark of the parallel product
    let start = Instant::now();
    let mut a = 0.;

    let mut f: Vec<f64> = (0..n)
        .map(|k| {
            let i = k / (nx + 1);
            let j = k % (nx + 1);
            let y = i as f64 / nx as f64;
            let x = j as f64 / ny as f64;
            y + x * x
        })
        .collect();
    zero_mean(&mut f);

    // largest eigenvalue by the power method
    for _iter in 0..nx.max(ny) {
        f = m.vec_mult(&f);
        zero_mean(&mut f);
        let nf: f64 = f.par_iter().map(|f| *f * *f).sum();
        let nf = (nf / n as f64).sqrt();
        a = nf;
        f.par_iter_mut().for_each(|f| *f /= nf);
    }

    let duration = start.elapsed();
    println!("Power method time: {:?} eig={}", duration, a);

    let xp: Vec<f64> = (0..nx + 1).map(|i| i as f64 * dx).collect();
    let yp: Vec<f64> = (0..ny + 1).map(|i| i as f64 * dy).collect();

    // convert to {-1,1} for get the nodal line

    f.par_iter_mut().for_each(|f| {
        *f = if *f > 0. { 1. } else { -1. };
    });
    println!("Plot nodal line");
    plotpy(xp, yp, f.clone());

    println!("OK");

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

#[allow(dead_code)]
fn apply_dirichlet(x: &mut Vec<f64>, nx: usize, ny: usize) {
    x.iter_mut().enumerate().for_each(|(k, x)| {
        let i = k % (nx + 1);
        let j = k / (nx + 1);
        if i == 0 || i == nx || j == 0 || j == ny {
            *x = 0.;
        }
    });
}

fn zero_mean(x: &mut Vec<f64>) {
    let m: f64 = x.iter().map(|x| *x as f64).sum::<f64>() / x.len() as f64;
    x.iter_mut().for_each(|x| {
        *x -= m;
    });
}

fn dirichlet(lx: f64, ly: f64, nx: usize, ny: usize) -> Vec<(usize, usize, f64)> {
    let dx = lx / nx as f64;
    let dy = ly / ny as f64;

    let mut vecval = vec![];

    let n = (nx + 1) * (ny + 1);
    for k in 0..n {
        let i = k % (nx + 1);
        let j = k / (nx + 1);
        if i == 0 || i == nx || j == 0 || j == ny {
            //vecval.push((k, k, 100. / dx / dx));
            vecval.push((k, k, 1e20));
        } else {
            vecval.push((k, k, 2. / dx / dx + 2. / dy / dy));
        }
    }

    for i in 0..nx {
        for j in 0..ny + 1 {
            let k1 = j * (nx + 1) + i;
            let k2 = j * (nx + 1) + i + 1;
            vecval.push((k1, k2, -1. / dx / dx));
            vecval.push((k2, k1, -1. / dx / dx));
        }
    }

    for j in 0..ny {
        for i in 0..nx + 1 {
            let k1 = j * (nx + 1) + i;
            let k2 = (j + 1) * (nx + 1) + i;
            vecval.push((k1, k2, -1. / dy / dy));
            vecval.push((k2, k1, -1. / dy / dy));
        }
    }

    vecval
}

fn lapgraph(nx: usize, ny: usize) -> Vec<(usize, usize, f64)> {
    let mut vecval = vec![];

    let n = (nx + 1) * (ny + 1);
    for k in 0..n {
        // let i = k % (nx + 1);
        // let j = k / (nx + 1);
        vecval.push((k, k, 4.));
    }

    for i in 0..nx {
        for j in 0..ny + 1 {
            let k1 = j * (nx + 1) + i;
            let k2 = j * (nx + 1) + i + 1;
            vecval.push((k1, k2, 1.));
            vecval.push((k2, k1, 1.));
        }
    }

    for j in 0..ny {
        for i in 0..nx + 1 {
            let k1 = j * (nx + 1) + i;
            let k2 = (j + 1) * (nx + 1) + i;
            vecval.push((k1, k2, 1.));
            vecval.push((k2, k1, 1.));
        }
    }

    vecval
}
