use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use polars::prelude::*;

pub fn polars_err(e: PolarsError) -> PyErr {
    PyRuntimeError::new_err(format!("Polars error: {}", e))
}

pub fn default_rolling_option(n:i64)-> RollingOptionsFixedWindow {
    RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        ..Default::default()
    }
} 

pub fn calculate_pearson_corr(s1: &Series, s2: &Series, n: i64) -> Result<Series, PolarsError> {
    let rolling_opts = default_rolling_option(n);
    let s1_mean = s1.rolling_mean(rolling_opts.clone())?;
    let s2_mean = s2.rolling_mean(rolling_opts.clone())?;
    let diff1 = (s1 - &s1_mean)?;
    let diff2 = (s2 - &s2_mean)?;
    let prod = (&diff1 * &diff2)?;
    let cov = prod.rolling_mean(rolling_opts.clone())?;

    let s1_std = s1.rolling_std(rolling_opts.clone())?;
    let s2_std = s2.rolling_std(rolling_opts)?;

    // Calculate correlation: cov / (std1 * std2)
    let std_prod = (&s1_std * &s2_std)?;
    let result = (&cov / &std_prod)?;

    Ok(result)
}