use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3_polars::types::PySeries;
use polars::prelude::*;

#[pyfunction]
fn ts_delay(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let out: Series = data.into();
    let out = out.shift(n);
    Ok(PySeries(out))
}

#[pyfunction]
fn ts_delta(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let out: Series = data.into();
    let delayed_data = out.shift(n);
    let delta = (&out - &delayed_data).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(delta))
}

#[pyfunction]
fn ts_returns(
    data: PySeries,
    n: i64,
) -> PyResult<PySeries> {
    let out: Series = data.into();
    let delayed_data = out.shift(n);
    let delta = (&out - &delayed_data).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;

    let returns = (&delta / &delayed_data).map_err(|e| {
        PyRuntimeError::new_err(format!("Polars error: {}", e))
    })?;
    Ok(PySeries(returns))
}

#[pymodule]
pub fn operators(m: &Bound<'_, PyModule>)-> PyResult<()>{
    m.add_function(wrap_pyfunction!(ts_delay, m)?)?;
    m.add_function(wrap_pyfunction!(ts_delta, m)?)?;    
    m.add_function(wrap_pyfunction!(ts_returns, m)?)?;    
    Ok(())
}