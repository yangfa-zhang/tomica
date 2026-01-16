use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3_polars::types::PySeries;
use polars::prelude::*;
use polars_compute::rolling::RollingQuantileParams;

mod utils;
use utils::*;

#[pymodule]
pub fn operators(m: &Bound<'_, PyModule>)-> PyResult<()>{
    // Time series operators
    m.add_function(wrap_pyfunction!(ts_delay, m)?)?;
    m.add_function(wrap_pyfunction!(ts_delta, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_returns, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_sum, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_product, m)?)?;   
    m.add_function(wrap_pyfunction!(ts_max, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_min, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_mean, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_std, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_rank, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_variance, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_quantile_up, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_quantile_down, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_zscore, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_robust_zscore, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_scale, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_sharpe, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_av_diff, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_min_diff, m)?)?; 
    m.add_function(wrap_pyfunction!(ts_max_diff, m)?)?;
    m.add_function(wrap_pyfunction!(ts_cov, m)?)?;
    m.add_function(wrap_pyfunction!(ts_beta, m)?)?;
    m.add_function(wrap_pyfunction!(ts_corr, m)?)?;
    m.add_function(wrap_pyfunction!(ts_regression, m)?)?;
    m.add_function(wrap_pyfunction!(ts_pos_count, m)?)?;
    m.add_function(wrap_pyfunction!(ts_neg_count, m)?)?;
    m.add_function(wrap_pyfunction!(ts_decay, m)?)?;
    m.add_function(wrap_pyfunction!(ts_argmax, m)?)?;
    m.add_function(wrap_pyfunction!(ts_argmin, m)?)?;
    // Cross section operators
    m.add_function(wrap_pyfunction!(cs_rank, m)?)?;
    Ok(())
}



#[pyfunction]
fn ts_delay(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.shift(n);
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_delta(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let delayed_data = data.shift(n);
    let result = (&data - &delayed_data).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_returns(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let delayed_data = data.shift(n);
    let delta = (&data - &delayed_data).map_err(polars_err)?;

    let result = (&delta / &delayed_data).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_sum(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries>{
    let data: Series = data.into();
    let result = data.rolling_sum(default_rolling_option(n)).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_product(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_sum(default_rolling_option(n)).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_max(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_max(default_rolling_option(n)).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_min(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_min(default_rolling_option(n)).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_mean(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_mean(default_rolling_option(n)).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_std(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_std(default_rolling_option(n)).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_rank(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_rank(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        fn_params: Some(RollingFnParams::Rank{
            method:RollingRankMethod::Average,
            seed: Some(42),
        }),
        ..Default::default()
    }).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_variance(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_var(default_rolling_option(n)).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_quantile_up(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries>{
    let data: Series = data.into();
    let result = data.rolling_quantile(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        fn_params: Some(
            RollingFnParams::Quantile(
                RollingQuantileParams {
                    prob: 0.75,
                    method: QuantileMethod::Nearest,
                }
            )
        ),
        ..Default::default()
    }).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_quantile_down(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries>{
    let data: Series = data.into();
    let result = data.rolling_quantile(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        fn_params: Some(
            RollingFnParams::Quantile(
                RollingQuantileParams {
                    prob: 0.25,
                    method: QuantileMethod::Nearest,
                }
            )
        ),
        ..Default::default()
    }).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_zscore(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries>{
    let data: Series = data.into();
    let rolling_opts = default_rolling_option(n);
    let data_mean = data.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let data_std = data.rolling_std(rolling_opts).map_err(polars_err)?;
    let diff = (&data - &data_mean).map_err(polars_err)?;
    let result = (&diff/&data_std).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_robust_zscore(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();

    let rolling_opts = RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,
        center: false,
        fn_params: Some(
            RollingFnParams::Quantile(
                RollingQuantileParams {
                    prob: 0.50,
                    method: QuantileMethod::Nearest,
                }
            )
        ),
        ..Default::default()
    };
    let med = data
        .rolling_quantile(rolling_opts.clone())
        .map_err(polars_err)?;
    let diff: Series = (&data - &med)
        .map_err(polars_err)?;
    let abs_diff = match diff.dtype() {
        DataType::Int32 => diff.i32().unwrap()
            .apply(|opt_v| opt_v.map(|v| v.abs()))
            .into_series(),
        DataType::Int64 => diff.i64().unwrap()
            .apply(|opt_v| opt_v.map(|v| v.abs()))
            .into_series(),
        DataType::Float32 => diff.f32().unwrap()
            .apply(|opt_v| opt_v.map(|v| v.abs()))
            .into_series(),
        DataType::Float64 => diff.f64().unwrap()
            .apply(|opt_v| opt_v.map(|v| v.abs()))
            .into_series(),
        _ => return Err(PyRuntimeError::new_err("unsupported dtype for abs")),
    };
    let mad = abs_diff.rolling_quantile(rolling_opts).map_err(polars_err)?;
    let mad_scaled = &mad * 1.4826;
    let result = (&diff / &mad_scaled).map_err(polars_err)?;

    Ok(PySeries(result))
}

#[pyfunction]
fn ts_scale(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let rolling_opts = default_rolling_option(n);
    let data_min = data.rolling_min(rolling_opts.clone()).map_err(polars_err)?;
    let data_max = data.rolling_max(rolling_opts).map_err(polars_err)?;
    let diff1 = (&data - &data_min).map_err(polars_err)?;
    let diff2 = (&data_max - &data_min).map_err(polars_err)?;
    let result = (&diff1/&diff2).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_sharpe(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let rolling_opts = default_rolling_option(n);
    let data_mean = data.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let data_std = data.rolling_std(rolling_opts).map_err(polars_err)?;
    let result = (&data_mean/&data_std).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_av_diff(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let data_mean = data.rolling_mean(default_rolling_option(n)).map_err(polars_err)?;
    let result = (&data - &data_mean).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_min_diff(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let data_min = data.rolling_min(default_rolling_option(n)).map_err(polars_err)?;
    let result = (&data - &data_min).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_max_diff(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let data_max = data.rolling_max(default_rolling_option(n)).map_err(polars_err)?;
    let result = (&data - &data_max).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_cov(data1: PySeries, data2: PySeries, n: i64) -> PyResult<PySeries> {
    let s1: Series = data1.into();
    let s2: Series = data2.into();

    if s1.len() != s2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Series lengths must match",
        ));
    }
    let rolling_opts = default_rolling_option(n);
    let s1_mean = s1.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let s2_mean = s2.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let diff1 = (&s1-&s1_mean).map_err(polars_err)?;
    let diff2 = (&s2-&s2_mean).map_err(polars_err)?;
    let prod = (&diff1 * &diff2).map_err(polars_err)?;
    let result = prod.rolling_mean(rolling_opts).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_beta(data1: PySeries, data2: PySeries, n: i64) -> PyResult<PySeries> {
    let s1: Series = data1.into();
    let s2: Series = data2.into();

    if s1.len() != s2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Series lengths must match",
        ));
    }
    let rolling_opts = default_rolling_option(n);
    let s1_mean = s1.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let s2_mean = s2.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let diff1 = (&s1-&s1_mean).map_err(polars_err)?;
    let diff2 = (&s2-&s2_mean).map_err(polars_err)?;
    let prod = (&diff1 * &diff2).map_err(polars_err)?;
    let s1_s2_cov = prod.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let s1_var = s1.rolling_var(rolling_opts).map_err(polars_err)?;
    let result = (&s1_s2_cov/&s1_var).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_corr(data1: PySeries, data2: PySeries, n: i64, rank: Option<bool>) -> PyResult<PySeries> {
    let s1: Series = data1.into();
    let s2: Series = data2.into();

    if s1.len() != s2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Series lengths must match",
        ));
    }

    let use_rank = rank.unwrap_or(false);

    if use_rank {
        let rolling_opts = RollingOptionsFixedWindow {
            window_size: n as usize,
            min_periods: 1,
            center: false,
            fn_params: Some(RollingFnParams::Rank {
                method: RollingRankMethod::Average,
                seed: Some(42),
            }),
            ..Default::default()
        };

        let s1_rank = s1.rolling_rank(rolling_opts.clone()).map_err(polars_err)?;
        let s2_rank = s2.rolling_rank(rolling_opts).map_err(polars_err)?;
        let result = calculate_pearson_corr(&s1_rank, &s2_rank, n).map_err(polars_err)?;
        Ok(PySeries(result))
    } else {
        let result = calculate_pearson_corr(&s1, &s2, n).map_err(polars_err)?;
        Ok(PySeries(result))
    }
}

#[pyfunction]
fn ts_regression(data1: PySeries, data2: PySeries, n: i64, rettype: i64) -> PyResult<PySeries> {
    // Returns results of linear model y = beta * x + alpha + resid
    // data1:   features (x)
    // data2:   label (y)
    // n:       period
    // rettype: return type. 0 for resid, 1 for beta, 2 for alpha, 3 for y_hat, 4 for R^2
    let s1: Series = data1.into();
    let s2: Series = data2.into();

    if s1.len() != s2.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Series lengths must match",
        ));
    }

    // Validate rettype is in valid range [0, 4]
    if !(0..=4).contains(&rettype) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            format!("rettype must be 0-4, got {}", rettype),
        ));
    }

    let rolling_opts = default_rolling_option(n);
    let s1_mean = s1.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let s2_mean = s2.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let diff1 = (&s1 - &s1_mean).map_err(polars_err)?;
    let diff2 = (&s2 - &s2_mean).map_err(polars_err)?;
    let prod = (&diff1 * &diff2).map_err(polars_err)?;
    let cov = prod.rolling_mean(rolling_opts.clone()).map_err(polars_err)?;
    let s1_var = s1.rolling_var(rolling_opts).map_err(polars_err)?;
    let beta = (&cov / &s1_var).map_err(polars_err)?;
    match rettype {
        0 => {
            // Return residuals: resid = y - y_hat
            let beta_s1_mean = (&beta * &s1_mean).map_err(polars_err)?;
            let alpha = (&s2_mean - &beta_s1_mean).map_err(polars_err)?;
            let beta_s1 = (&beta * &s1).map_err(polars_err)?;
            let predict = (&beta_s1 + &alpha).map_err(polars_err)?;
            let resid = (&s2 - &predict).map_err(polars_err)?;
            Ok(PySeries(resid))
        }
        1 => {
            Ok(PySeries(beta))
        }
        2 => {
            // Return alpha: alpha = mean(y) - beta * mean(x)
            let beta_s1_mean = (&beta * &s1_mean).map_err(polars_err)?;
            let alpha = (&s2_mean - &beta_s1_mean).map_err(polars_err)?;
            Ok(PySeries(alpha))
        }
        3 => {
            // Return y_hat: y_hat = beta * x + alpha
            let beta_s1_mean = (&beta * &s1_mean).map_err(polars_err)?;
            let alpha = (&s2_mean - &beta_s1_mean).map_err(polars_err)?;
            let beta_s1 = (&beta * &s1).map_err(polars_err)?;
            let predict = (&beta_s1 + &alpha).map_err(polars_err)?;
            Ok(PySeries(predict))
        }
        4 => {
            // Return R^2: rank correlation between y_hat and y
            let beta_s1_mean = (&beta * &s1_mean).map_err(polars_err)?;
            let alpha = (&s2_mean - &beta_s1_mean).map_err(polars_err)?;
            let beta_s1 = (&beta * &s1).map_err(polars_err)?;
            let predict = (&beta_s1 + &alpha).map_err(polars_err)?;

            let rolling_rank_opts = RollingOptionsFixedWindow {
                window_size: n as usize,
                min_periods: 1,
                center: false,
                fn_params: Some(RollingFnParams::Rank {
                    method: RollingRankMethod::Average,
                    seed: Some(42),
                }),
                ..Default::default()
            };

            let predict_rank = predict.rolling_rank(rolling_rank_opts.clone()).map_err(polars_err)?;
            let s2_rank = s2.rolling_rank(rolling_rank_opts).map_err(polars_err)?;
            let r_squared = calculate_pearson_corr(&predict_rank, &s2_rank, n).map_err(polars_err)?;
            Ok(PySeries(r_squared))
        }
        _ => unreachable!("rettype validation should prevent this"),
    }
}

#[pyfunction]
fn ts_pos_count(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();

    let pos = data
        .gt(0).map_err(polars_err)?         
        .cast(&DataType::Int64).map_err(polars_err)?;  
    let result = pos.rolling_sum(default_rolling_option(n)).map_err(polars_err)?;

    Ok(PySeries(result))
}

#[pyfunction]
fn ts_neg_count(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();

    let pos = data
        .lt(0).map_err(polars_err)?         
        .cast(&DataType::Int64).map_err(polars_err)?;  
    let result = pos.rolling_sum(default_rolling_option(n)).map_err(polars_err)?;

    Ok(PySeries(result))
}

#[pyfunction]
fn ts_decay(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let n_usize = n as usize;
    let mut weights: Vec<f64> = (1..=n_usize).map(|i| i as f64).collect();
    let total: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }
    let rolling_opts = RollingOptionsFixedWindow {
        window_size: n_usize,
        min_periods: 1,   
        center: false,
        weights: Some(weights),
        ..Default::default()
    };
    let result = data.rolling_mean(rolling_opts).map_err(polars_err)?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_argmax(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let n_usize = n as usize;
    let len = data.len();
    let data_f64 = data.cast(&DataType::Float64).map_err(polars_err)?;
    let ca = data_f64.f64().map_err(polars_err)?;

    // Calculate argmax for each rolling window
    let mut result_vec: Vec<Option<f64>> = Vec::with_capacity(len);

    for i in 0..len {
        if i < n_usize - 1 {
            result_vec.push(None);
        } else {
            let start = i + 1 - n_usize;
            let mut max_val = f64::NEG_INFINITY;
            let mut max_idx: i64 = 0;

            for j in 0..n_usize {
                if let Some(val) = ca.get(start + j) {
                    if val > max_val {
                        max_val = val;
                        max_idx = j as i64;
                    }
                }
            }

            if max_val.is_finite() {
                result_vec.push(Some(max_idx as f64));
            } else {
                result_vec.push(None);
            }
        }
    }

    let result = Series::new("argmax".into(), result_vec);
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_argmin(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let n_usize = n as usize;
    let len = data.len();
    let data_f64 = data.cast(&DataType::Float64).map_err(polars_err)?;
    let ca = data_f64.f64().map_err(polars_err)?;

    // Calculate argmin for each rolling window
    let mut result_vec: Vec<Option<f64>> = Vec::with_capacity(len);

    for i in 0..len {
        if i < n_usize - 1 {
            result_vec.push(None);
        } else {
            let start = i + 1 - n_usize;
            let mut min_val = f64::INFINITY;
            let mut min_idx: i64 = 0;

            for j in 0..n_usize {
                if let Some(val) = ca.get(start + j) {
                    if val < min_val {
                        min_val = val;
                        min_idx = j as i64;
                    }
                }
            }

            if min_val.is_finite() {
                result_vec.push(Some(min_idx as f64));
            } else {
                result_vec.push(None);
            }
        }
    }

    let result = Series::new("argmin".into(), result_vec);
    Ok(PySeries(result))
}

#[pyfunction]
fn cs_rank(data: PySeries, pct: Option<bool>) -> PyResult<PySeries> {
    // Cross-sectional rank: ranks values and optionally returns percentile (0.0 to 1.0)
    let data: Series = data.into();
    let use_pct = pct.unwrap_or(true);

    // Convert to f64 for ranking
    let data_f64 = data.cast(&DataType::Float64).map_err(polars_err)?;
    let ca = data_f64.f64().map_err(polars_err)?;

    let len = ca.len();
    let mut result_vec: Vec<Option<f64>> = Vec::with_capacity(len);

    // Collect non-null values with their indices
    let mut values_with_idx: Vec<(usize, f64)> = Vec::new();
    for (idx, opt_val) in ca.iter().enumerate() {
        if let Some(val) = opt_val {
            values_with_idx.push((idx, val));
        }
    }

    let valid_count = values_with_idx.len();

    if valid_count == 0 {
        return Ok(PySeries(Series::new("".into(), vec![Option::<f64>::None; len])));
    }
    values_with_idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks: Vec<(usize, f64)> = Vec::with_capacity(valid_count);
    let mut i = 0;
    while i < valid_count {
        let current_val = values_with_idx[i].1;
        let mut j = i;
        while j < valid_count && values_with_idx[j].1 == current_val {
            j += 1;
        }

        // Average rank for tied values (1-based indexing)
        let avg_rank = ((i + 1 + j) as f64) / 2.0;

        for k in i..j {
            ranks.push((values_with_idx[k].0, avg_rank));
        }

        i = j;
    }
    for _ in 0..len {
        result_vec.push(None);
    }

    for (idx, rank) in ranks {
        if use_pct {
            // Convert to percentile: (rank - 1) / (valid_count - 1)
            if valid_count > 1 {
                result_vec[idx] = Some((rank - 1.0) / ((valid_count - 1) as f64));
            } else {
                result_vec[idx] = Some(0.0);
            }
        } else {
            result_vec[idx] = Some(rank);
        }
    }

    let result = Series::new("".into(), result_vec);
    Ok(PySeries(result))
}
