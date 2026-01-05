use pyo3::prelude::*;
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

#[pymodule]
pub fn operators(m: &Bound<'_, PyModule>)-> PyResult<()>{
    m.add_function(wrap_pyfunction!(ts_delay, m)?)?;
    Ok(())
}