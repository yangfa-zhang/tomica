#![allow(non_snake_case)]
use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;

mod operators;

#[pymodule]
mod tomica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}