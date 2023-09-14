use ndarray::{parallel::prelude::*, Array1, Array2};
use rayon::prelude::*;

pub fn argsort(data: &[f64]) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<_>>();
    indices.par_sort_by(|&i1, &i2| data[i1].partial_cmp(&data[i2]).unwrap());
    indices
}

pub fn hq(cumhist: &Array2<i32>, cumhist_log: &Array2<f64>, q: usize, p: usize, n: usize) -> f64 {
    let total: f64 = n as f64;
    let total_log = total.ln();

    (0..q).fold(0.0, |acc, i| {
        acc - (cumhist[[i, p - 1]] as f64 / total) * (cumhist_log[[i, p - 1]] - total_log)
    })
}

pub fn hp3(c: &Array1<usize>, c_log: &Array1<f64>, s: usize, t: usize) -> f64 {
    if s == t {
        return 0.0;
    }

    let total = c[t - 1] as f64;
    let total_log = total.ln();

    let prob_s = c[s - 1] as f64 / total;
    let prob_log_s = c_log[s - 1] - total_log;

    let sum = c[t - 1] - c[s - 1];
    let prob_t = sum as f64 / total;
    let prob_log_t = if sum != 0 {
        (sum as f64).ln() - total_log
    } else {
        0.0
    };

    -(prob_s * prob_log_s + prob_t * prob_log_t)
}

pub fn hp3q(
    cumhist: &Array2<i32>,
    cumhist_log: &Array2<f64>,
    c: &Array1<usize>,
    q: usize,
    s: usize,
    t: usize,
) -> f64 {
    let total = c[t - 1] as f64;
    let total_log = total.ln();

    (0..q).fold(0.0, |acc, i| {
        let prob_s = cumhist[[i, s - 1]] as f64 / total;
        let prob_t = cumhist[[i, t - 1]] as f64 / total;

        let prob_log_s = cumhist_log[[i, s - 1]] - total_log;
        let prob_log_t = cumhist_log[[i, t - 1]] - total_log;

        acc - (prob_s * prob_log_s + (prob_t - prob_s) * prob_log_t)
    })
}

pub fn hp2q(cumhist: &Array2<i32>, c: &Array1<usize>, q: usize, s: usize, t: usize) -> f64 {
    if s == t {
        return 0.0;
    }

    let total = (c[t - 1] - c[s - 1]) as f64;
    let total_log = total.ln();

    (0..q).fold(0.0, |acc, i| {
        let sum = cumhist[[i, t - 1]] - cumhist[[i, s - 1]];
        let prob = sum as f64 / total;
        let prob_log = if sum != 0 {
            (sum as f64).ln() - total_log
        } else {
            0.0
        };
        acc - prob * prob_log
    })
}

pub fn equipartition_y_axis(
    dy: &Array1<f64>,
    y: usize,
    q_map: &Array1<i32>,
) -> (Array1<i32>, usize) {
    let n = dy.len();
    let mut i = 0;
    let mut h = 0;
    let mut curr = 0;
    let mut rowsize = n as f64 / y as f64;
    let mut q_map = q_map.to_owned();

    while i < n {
        let s = (i..n).take_while(|&j| dy[i] == dy[j]).count();

        let temp1 = (h as f64 + s as f64 - rowsize).abs();
        let temp2 = (h as f64 - rowsize).abs();

        if h != 0 && temp1 >= temp2 {
            curr += 1;
            h = 0;
            let temp1 = n as f64 - i as f64;
            let temp2 = y as f64 - curr as f64;
            rowsize = temp1 / temp2;
        }

        for j in 0..s {
            q_map[i + j] = curr;
        }

        i += s;
        h += s;
    }

    let q = (curr + 1) as usize;
    (q_map, q)
}

pub fn get_clumps_partition(dx: &Array1<f64>, q_map: &Array1<i32>) -> Option<(Array1<i32>, usize)> {
    let n = dx.len();
    let mut c: i32 = -1;
    let mut q_tilde = q_map.to_owned();
    let mut p_map = Array1::zeros(n);

    let mut i = 0;
    while i < n {
        let mut s = 1;
        let mut flag = false;
        let mut j = i + 1;

        while j < n {
            if dx[i] == dx[j] {
                if q_tilde[i] != q_tilde[j] {
                    flag = true;
                }
                s += 1;
                j += 1;
            } else {
                break;
            }
        }

        if s > 1 && flag {
            for k in i..i + s {
                q_tilde[k] = c;
            }
            c -= 1;
        }

        i += s;
    }

    p_map[0] = 0;
    let mut i = 0;
    for j in 1..n {
        if q_tilde[j] != q_tilde[j - 1] {
            i += 1;
        }
        p_map[j] = i;
    }

    let p = (i + 1) as usize;
    Some((p_map, p))
}

pub fn get_superclumps_partition(
    dx: &Array1<f64>,
    k_hat: usize,
    q_map: &Array1<i32>,
) -> Option<(Array1<i32>, usize)> {
    // Clumps
    let (mut p_map, mut p) = get_clumps_partition(dx, q_map)?;

    // Superclumps
    if p > k_hat {
        let dp: Array1<f64> = p_map.iter().map(|&x| x as f64).collect();
        (p_map, p) = equipartition_y_axis(&dp, k_hat, &p_map);
    }

    Some((p_map, p))
}

pub fn compute_c(p_map: &Array1<i32>, p: usize, n: usize) -> Option<Array1<usize>> {
    let mut c = Array1::zeros(p);

    for i in 0..n {
        c[p_map[i] as usize] += 1;
    }

    for i in 1..p {
        c[i] += c[i - 1];
    }

    Some(c)
}

pub fn compute_c_log(c: &Array1<usize>, p: usize) -> Option<Array1<f64>> {
    let mut c_log = Array1::zeros(p);

    for i in 0..p {
        if c[i] != 0 {
            c_log[i] = (c[i] as f64).ln();
        }
    }

    Some(c_log)
}

pub fn compute_cumhist(
    q_map: &Array1<i32>,
    q: usize,
    p_map: &Array1<i32>,
    p: usize,
    n: usize,
) -> Option<Array2<i32>> {
    let mut cumhist = Array2::zeros((q, p));

    if q_map.iter().any(|&q_idx| q_idx < 0 || q_idx as usize >= q)
        || p_map.iter().any(|&p_idx| p_idx < 0 || p_idx as usize >= p)
    {
        return None; // Invalid index
    }

    (0..n).for_each(|i| {
        let q_index = q_map[i] as usize;
        let p_index = p_map[i] as usize;
        cumhist[[q_index, p_index]] += 1;
    });

    (0..q).for_each(|q_index| {
        (1..p).for_each(|p_index| {
            cumhist[[q_index, p_index]] += cumhist[[q_index, p_index - 1]];
        });
    });

    Some(cumhist)
}

pub fn compute_cumhist_log(cumhist: &Array2<i32>, q: usize, p: usize) -> Option<Array2<f64>> {
    let cumhist_log: Array1<f64> = cumhist
        .iter()
        .map(|&value| {
            if value != 0 {
                (value as f64).ln()
            } else {
                0.0 // Logarithm of 0 is undefined, so we'll treat it as 0
            }
        })
        .collect();

    cumhist_log.into_shape((q, p)).ok()
}

pub fn init_i(p: usize, x: usize) -> Option<Array2<f64>> {
    Some(Array2::zeros((p + 1, x + 1)))
}

pub fn compute_hp2q(
    cumhist: &Array2<i32>,
    c: &Array1<usize>,
    q: usize,
    p: usize,
) -> Option<Array2<f64>> {
    let mut m = Array2::zeros((p + 1, p + 1));

    for t in 3..=p {
        for s in 2..=t {
            m[[s, t]] = hp2q(cumhist, c, q, s, t);
        }
    }

    Some(m)
}

pub fn optimize_x_axis(
    n: usize,
    q_map: &Array1<i32>,
    q: usize,
    p_map: &Array1<i32>,
    p: usize,
    x: usize,
    score: &mut [f64],
) {
    if p == 1 {
        score.iter_mut().take(x - 1).for_each(|i| *i = 0.0);
    }

    let c = compute_c(p_map, p, n).unwrap();
    let c_log = compute_c_log(&c, p).unwrap();
    let cumhist = compute_cumhist(q_map, q, p_map, p, n).unwrap();
    let cumhist_log = compute_cumhist_log(&cumhist, q, p).unwrap();
    let hp2q = compute_hp2q(&cumhist, &c, q, p).unwrap();
    let mut i_matrix = init_i(p, x).unwrap();

    let hq = hq(&cumhist, &cumhist_log, q, p, n);

    for t in 2..=p {
        let f_max = (1..=t)
            .map(|s| hp3(&c, &c_log, s, t) - hp3q(&cumhist, &cumhist_log, &c, q, s, t))
            .max_by(|f1, f2| f1.partial_cmp(f2).unwrap_or(std::cmp::Ordering::Equal));

        if let Some(f) = f_max {
            i_matrix[[t, 2]] = hq + f;
        }
    }

    for l in 3..=x {
        for t in l..=p {
            let ct = c[t - 1] as f64;
            let f_max = ((l - 1)..=t)
                .map(|s| {
                    let cs = c[s - 1] as f64;

                    (cs / ct) * (i_matrix[[s, l - 1]] - hq) - ((ct - cs) / ct) * hp2q[[s, t]]
                })
                .max_by(|f1, f2| f1.partial_cmp(f2).unwrap_or(std::cmp::Ordering::Equal));

            if let Some(f) = f_max {
                i_matrix[[t, l]] = hq + f;
            }
        }
    }

    for i in (p + 1)..=x {
        i_matrix[[p, i]] = i_matrix[[p, p]];
    }

    for (i, log_i) in (2..=x).zip((2..=x).map(|i| f64::ln(i as f64))) {
        let log_q = f64::ln(q as f64);
        let min_log = log_i.min(log_q);
        score[i - 2] = i_matrix[[p, i]] / min_log;
    }
}

#[derive(Debug, Clone)]
pub struct MineProblem {
    pub n: usize,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub score: MineScore,
}

impl MineProblem {
    pub fn new(x: Vec<f64>, y: Vec<f64>, param: &MineParameter) -> Option<Self> {
        assert_eq!(x.len(), y.len());
        let n = x.len();
        let b = match param.alpha {
            alpha if alpha > 0.0 && alpha <= 1.0 => f64::max((n as f64).powf(alpha), 4.0),
            alpha if alpha >= 4.0 => f64::min(alpha, n as f64),
            _ => return None,
        };

        let score_n = f64::max((b / 2.0).floor(), 2.0) as usize - 1;
        let mut m = Vec::with_capacity(n);
        let mut mat = Vec::with_capacity(n);

        for i in 0..score_n {
            m.push((b / (i as f64 + 2.0)).floor() as usize - 1);
            mat.push(vec![0.0; m[i]]);
        }

        let score = MineScore { n: score_n, m, mat };

        Some(Self { n, x, y, score })
    }
}

#[derive(Debug, Clone)]
pub struct MineParameter {
    pub alpha: f64,
    pub c: f64,
}

impl MineParameter {
    pub fn new(alpha: f64, c: f64) -> Self {
        Self { alpha, c }
    }
}

#[derive(Clone, Debug)]
pub struct MineScore {
    pub n: usize,
    pub m: Vec<usize>,
    pub mat: Vec<Vec<f64>>,
}

pub fn mine_compute_score(prob: &MineProblem, param: &MineParameter) -> Option<MineScore> {
    let mut score = prob.score.clone();
    let q_map = Array1::zeros(prob.n);
    let mut xx = Array1::zeros(prob.n);
    let mut yy = Array1::zeros(prob.n);
    let mut xy = Array1::zeros(prob.n);
    let mut yx = Array1::zeros(prob.n);
    let ix = argsort(&prob.x);
    let iy = argsort(&prob.y);
    for i in 0..prob.n {
        xx[i] = prob.x[ix[i]];
        yy[i] = prob.y[iy[i]];
        xy[i] = prob.x[iy[i]];
        yx[i] = prob.y[ix[i]];
    }

    // /* x vs. y */
    score.mat.par_iter_mut().enumerate().for_each(|(i, row)| {
        let k = usize::max((param.c * (score.m[i] as f64 + 1.0)) as usize, 1);

        let (mut q_map, q) = equipartition_y_axis(&yy, i + 2, &q_map);

        // Sort Q by x
        for j in 0..prob.n {
            q_map[ix[j]] = q_map[j];
        }

        let (p_map, p) = get_superclumps_partition(&xx, k, &q_map).unwrap();

        optimize_x_axis(
            prob.n,
            &q_map,
            q,
            &p_map,
            p,
            usize::min(i + 2, score.m[i] + 1),
            row,
        );
    });

    /* y vs. x */
    score.mat.par_iter_mut().enumerate().for_each(|(i, row)| {
        let k = usize::max((param.c * (score.m[i] as f64 + 1.0)) as usize, 1);

        let (mut q_map, q) = equipartition_y_axis(&xx, i + 2, &q_map);

        // Sort Q by x
        for j in 0..prob.n {
            q_map[iy[j]] = q_map[j];
        }

        let (p_map, p) = get_superclumps_partition(&yy, k, &q_map).unwrap();

        optimize_x_axis(
            prob.n,
            &q_map,
            q,
            &p_map,
            p,
            usize::min(i + 2, score.m[i] + 1),
            row,
        );
    });

    Some(score)
}

pub fn mine_mic(score: &MineScore) -> f64 {
    let mut score_max = 0.0;

    for i in 0..score.n {
        for j in 0..score.m[i] {
            if score.mat[i][j] > score_max {
                score_max = score.mat[i][j];
            }
        }
    }

    score_max
}

pub fn mine_mas(score: &MineScore) -> f64 {
    let mut score_max = 0.0;

    for i in 0..score.n {
        for j in 0..score.m[i] {
            let score_curr = (score.mat[i][j] - score.mat[j][i]).abs();
            if score_curr > score_max {
                score_max = score_curr;
            }
        }
    }

    score_max
}

pub fn mine_mev(score: &MineScore) -> f64 {
    let mut score_max = 0.0;

    for i in 0..score.n {
        for j in 0..score.m[i] {
            if (j == 0 || i == 0) && score.mat[i][j] > score_max {
                score_max = score.mat[i][j];
            }
        }
    }

    score_max
}

pub fn mine_mcn(score: &MineScore, eps: f64) -> f64 {
    let mut score_min = f64::MAX;
    let delta = 0.0001; // avoids overestimation of mcn
    let mic = mine_mic(score);

    for i in 0..score.n {
        for j in 0..score.m[i] {
            let log_xy = ((i as f64 + 2.0) * (j as f64 + 2.0)).ln() / 2.0_f64.ln();
            if ((score.mat[i][j] + delta) >= ((1.0 - eps) * mic)) && (log_xy < score_min) {
                score_min = log_xy;
            }
        }
    }

    score_min
}

pub fn mine_mcn_general(score: &MineScore) -> f64 {
    let mut log_xy: f64;
    let mut score_min: f64 = f64::MAX;
    let delta: f64 = 0.0001; // avoids overestimation of mcn
    let mic: f64 = mine_mic(score);

    for i in 0..score.n {
        for j in 0..score.m[i] {
            log_xy = ((i as f64 + 2.0) * (j as f64 + 2.0)).log2();
            if (score.mat[i][j] + delta) >= (mic * mic) && log_xy < score_min {
                score_min = log_xy;
            }
        }
    }

    score_min
}

pub fn mine_tic(score: &MineScore, norm: bool) -> f64 {
    let mut tic = 0.0;
    let mut k = 0;

    for i in 0..score.n {
        for j in 0..score.m[i] {
            tic += score.mat[i][j];
            k += 1;
        }
    }

    if norm {
        tic /= k as f64;
    }

    tic
}
