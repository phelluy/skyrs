/// Some experiments with graph partitioning
fn main() {
    // grid definition
    let nx = 30;
    let ny = 60;

    println!("Assembling...");
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
    let mut m = skyrs::Sky::new(vecval.clone());

    // use rand::{thread_rng, Rng};

    // let mut rng = thread_rng();

    //let num: f64 = rng.gen_range(-40.0..1.3e5);

    let mut u: Vec<f64> = (0..n).map(|i| i as f64).collect();
    //let mut u: Vec<f64> = (0..n).map(|_i| rng.gen_range(0.0..1.)).collect();

    let mut lambda = 0.;

    for _iter in 0..1 {
        //let mut v = m.vec_mult(&u);
        let moyu: f64 = u.iter().sum::<f64>() / n as f64;
        u.iter_mut().for_each(|u| *u -= moyu);
        let normu: f64 = u.iter().fold(0.0 / 0.0, |m, v| v.max(m));
        let mut v = m.solve(u).unwrap();
        let moyv: f64 = v.iter().sum::<f64>() / n as f64;
        v.iter_mut().for_each(|u| *u -= moyv);
        // let normu: f64 = u.iter().map(|u| u * u).sum::<f64>().sqrt();
        // let normv: f64 = v.iter().map(|u| u * u).sum::<f64>().sqrt();
        let normv: f64 = v.iter().fold(0.0 / 0.0, |m, v| v.max(m));
        // println!("normu={}", normv);
        lambda = normv / normu;
        v.iter_mut().for_each(|u| (*u /= normv));
        u = v;
    }

    println!("lambda={}", lambda);
    //let  = m.solve(vec![1.; n]).unwrap();

    // plot
    let xp: Vec<f64> = (0..nx + 1).map(|i| i as f64).collect();
    let yp: Vec<f64> = (0..ny + 1).map(|i| i as f64).collect();

    println!("OK");

    u.iter_mut()
        .for_each(|u| *u = if *u > 0. { 1. } else { -1. });

    println!("Trying to plot...");
    plotpy(xp, yp, u);

    // graph test

    use petgraph::graphmap::UnGraphMap;

    let graph = UnGraphMap::<_, f64>::from_edges(vecval);

    use petgraph::visit::Bfs;

    let k0 = 0 * (nx + 1) + 0;

    let mut bfs = Bfs::new(&graph, k0);

    let mut v: Vec<f64> = vec![0.; n];

    let mut count: usize = 0;
    v[0] = k0 as f64;
    count += 1;
    while let Some(visited) = bfs.next(&graph) {
        println!(" {}", visited);
        v[visited] = count as f64;
        count += 1;
    }
    let xp: Vec<f64> = (0..nx + 1).map(|i| i as f64).collect();
    let yp: Vec<f64> = (0..ny + 1).map(|i| i as f64).collect();
    v.iter_mut()
        .for_each(|v| *v = if *v < n as f64 / 2. { 1. } else { -1. });
    plotpy(xp, yp, v);

    use petgraph::dot::Dot;

    use std::fs::File;
    use std::io::BufWriter;
    use std::io::Write;
    {
        let meshfile = File::create("graph.dot").unwrap();
        let mut meshfile = BufWriter::new(meshfile); // create a buffer for faster writes...
                                                     //println!("{}", Dot::new(&graph));
        let output = format!("{}", Dot::new(&graph));
        writeln!(meshfile, "{}", output).unwrap();
    }
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
