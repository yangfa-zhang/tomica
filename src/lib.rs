#![allow(non_snake_case)]
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;

pub mod base;

pub mod alpha;
pub mod backtest;
pub mod data;
pub mod report;

#[pymodule]
mod tomica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    
    Ok(())
}
