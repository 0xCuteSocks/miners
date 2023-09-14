use miners::*;
use rand::{thread_rng, Rng};
use std::f64::consts::PI;
use std::time::Instant;

fn print_stats(score: &MineScore) {
    println!("MIC_e: {}", mine_mic(score));
    println!("MAS: {}", mine_mas(score));
    println!("MEV: {}", mine_mev(score));
    println!("MCN (eps=0): {}", mine_mcn(score, 0.0));
    println!("MCN (eps=1-MIC): {}", mine_mcn_general(score));
    println!("TIC: {}", mine_tic(score, false));
}

fn main() {
    let x = (0..100000).map(|i| i as f64).collect::<Vec<f64>>();
    let mut y = x
        .iter()
        .map(|&x| (10.0 * PI * x).sin() + x)
        .collect::<Vec<f64>>();
    let mut rng = thread_rng();
    y.iter_mut().for_each(|y| *y += rng.gen_range(0.0..50000.0)); // add some noise

    println!("x len {:?}, y len {:?}", x.len(), y.len());

    let param = MineParameter {
        alpha: 0.6,
        c: 15.0,
    };

    let prob = MineProblem::new(x, y, &param).unwrap();
    let now = Instant::now();
    let computed_score =
        mine_compute_score(&prob.clone(), &param.clone()).expect("Failed to compute MineScore");

    println!("time cost: {:?}", now.elapsed());
    // Access the computed score's attributes if needed
    // println!("Computed Score (n): {}", computed_score.n);
    // println!("Computed Score (m): {:?}", computed_score.m);
    // println!("Computed Score (M): {:?}", computed_score.mat);
    println!("With noise:\n");
    print_stats(&computed_score);
}
