use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3_polars::types::PySeries;
use polars::prelude::*;

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
    let result = (&data - &delayed_data).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_returns(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let delayed_data = data.shift(n);
    let delta = (&data - &delayed_data).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;

    let result = (&delta / &delayed_data).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_sum(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries>{
    let data: Series = data.into();
    let result = data.rolling_sum(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        ..Default::default()
    }).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_product(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_sum(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        ..Default::default()
    }).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_max(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_max(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        ..Default::default()
    }).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_min(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_min(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        ..Default::default()
    }).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_mean(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_mean(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        ..Default::default()
    }).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_std(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_std(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        ..Default::default()
    }).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
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
        ..Default::default()
    }).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}

#[pyfunction]
fn ts_variance(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let data: Series = data.into();
    let result = data.rolling_var(RollingOptionsFixedWindow {
        window_size: n as usize,
        min_periods: 1,   
        center: false,
        ..Default::default()
    }).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(result))
}


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
    Ok(())
}