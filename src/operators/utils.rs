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