
use skyrs::Sky;

fn main() {
    let n = 9;
    let mut coo = vec![];
    
    // Helper to add a full dense block
    let mut add_block = |r_start: usize, c_start: usize| {
        for i in 0..3 {
            for j in 0..3 {
                let val = if r_start + i == c_start + j { 4.0 } else { -1.0 };
                coo.push((r_start + i, c_start + j, val));
            }
        }
    };

    // Construct Lower Block Triangular Matrix (Same as previous test)
    // D1 (0,0)
    add_block(0, 0);
    // D2 (3,3)
    add_block(3, 3);
    // D3 (6,6)
    add_block(6, 6);
    // L21 (3,0)
    add_block(3, 0);
    // L32 (6,3)
    add_block(6, 3);

    let mut sky = Sky::new(coo);
    
    // We must call solve or coo_to_sky to build the profile/skyline structure
    // otherwise plot will show nothing or crash if empty
    // Let's force structure build
    let u = vec![1.0; n];
    let _ = sky.solve(u); 

    println!("Matrix profile built.");
    
    // Plot to a known location
    let path = sky.plot("debug_9x9", 200).unwrap();
    println!("Generated plot at: {}", path);
}
