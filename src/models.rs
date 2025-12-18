
trait Model {
    fn train(&mut self, X: Vec<Vec<f64>>, y: Vec<f64>, epochs: usize);
    fn predict(&self, X: Vec<Vec<f64>>) -> Vec<f64>;
    fn evaluate(&self, X: Vec<Vec<f64>>, y: Vec<f64>) -> f64;
}