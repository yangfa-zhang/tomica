use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3_polars::types::PySeries;
use polars::prelude::*;
use polars_compute::rolling::RollingQuantileParams;

mod utils;
use utils::*;

#[pymodule]
pub fn operators(m: &Bound<'_, PyModule>)-> PyResult<()>{
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
    m.add_function(wrap_pyfunction!(ts_pos_count, m)?)?;
    m.add_function(wrap_pyfunction!(ts_neg_count, m)?)?;
    m.add_function(wrap_pyfunction!(decay_n, m)?)?;
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
fn decay_n(
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