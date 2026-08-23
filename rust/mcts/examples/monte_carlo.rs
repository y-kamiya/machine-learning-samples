use rand::prelude::*;

fn simulate(max: i32) -> f32 {
    let mut count = 0;
    let mut rng = rand::rng();
    for _ in 0..max {
        let x = rng.random_range(0.0..=1.0);
        let y = rng.random_range(0.0..=1.0);
        let d = x * x + y * y;
        if d <= 1.0 {
            count += 1;
        }
    }
    count as f32 / max as f32 * 4.0
}

fn main() {
    for n in [100, 1000, 10000, 100000] {
        let output = simulate(n);
        println!("sample count {n}: {output}");
    }
}
