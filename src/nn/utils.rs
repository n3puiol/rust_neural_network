use std::ops::{Add, Mul};
use crate::mnist_dataset::MnistRecord;

pub struct DataSet {
    pub inputs: Vec<Vector>,
    pub targets: Vec<Vector>,
}

pub fn normalize(data: &Vec<MnistRecord>) -> DataSet {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for record in data {
        inputs.push(Vector::from(record.pixels.clone()));
        let mut target = vec![0.0; 10];
        target[record.label as usize] = 1.0;
        targets.push(Vector::from(target));
    }

    DataSet {
        inputs,
        targets,
    }
}

pub struct Vector {
    data: Vec<f64>,
}

impl Vector {
    pub(crate) fn from(data: Vec<f64>) -> Vector {
        Vector { data }
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<f64> {
        self.data.iter()
    }

    pub(crate) fn iter_mut(&mut self) -> std::slice::IterMut<f64> {
        self.data.iter_mut()
    }
}

impl Clone for Vector {
    fn clone(&self) -> Vector {
        Vector {
            data: self.data.clone(),
        }
    }
}

impl Add for Vector {
    type Output = Vector;

    fn add(self, other: Vector) -> Vector {
        if self.data.len() != other.data.len() {
            panic!("Vectors must have the same length");
        }
        let result_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x + y)
            .collect();

        Vector { data: result_data }
    }
}

impl Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, scalar: f64) -> Vector {
        let result_data: Vec<f64> = self.data.iter().map(|&x| x * scalar).collect();
        Vector { data: result_data }
    }
}

impl Mul for Vector {
    type Output = Vector;

    fn mul(self, other: Vector) -> Vector {
        if self.data.len() != other.data.len() {
            panic!("Vectors must have the same length");
        }
        let result_data: Vec<f64> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&x, &y)| x * y)
            .collect();
        Vector::from(result_data)
    }
}

impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl std::fmt::Debug for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}