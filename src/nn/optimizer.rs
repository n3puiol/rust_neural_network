pub(crate) trait Optimizer {
    fn optimize(&self);
}

pub struct AdamOptimizer;

impl Optimizer for AdamOptimizer {
    fn optimize(&self) {
        todo!()
    }
}