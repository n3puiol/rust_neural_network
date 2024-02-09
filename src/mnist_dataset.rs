use std::{error::Error};

pub struct MnistRecord {
    pub label: u8,
    pub pixels: Vec<f64>,
}

pub fn load_mnist_dataset(data_path: &str, nb_records: usize) -> Result<Vec<MnistRecord>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_reader(std::fs::File::open(data_path)?);

    let mut mnist_records = Vec::new();

    for result in rdr.records().take(nb_records) {
        let record = result?;
        let mut pixels = Vec::new();
        for pixel in record.iter().skip(1) {
            pixels.push(pixel.parse()?);
        }
        mnist_records.push(MnistRecord {
            label: record[0].parse()?,
            pixels,
        });
    }
    Ok(mnist_records)
}