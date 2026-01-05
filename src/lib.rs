#![allow(non_snake_case)]
use pyo3::prelude::*;
use pyo3::wrap_pymodule;
pub mod operators;

#[pymodule]
fn tomica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(operators::operators))?;
    Ok(())
}