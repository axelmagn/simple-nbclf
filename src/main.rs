extern crate nalgebra as na;
extern crate num;

use std::io::prelude::*;
use std::io;
use std::fs::File;
use std::path::Path;
use std::str::FromStr;
use std::string::ToString;
use std::env;
use num::traits::Zero;
use na::{DMat, DVec};

struct DataMatrix<T> {
    data: Vec<T>,
    shape: (usize, usize),
}


impl<T: FromStr + ToString + Zero + Clone + Copy> DataMatrix<T> {
    fn open<P: AsRef<Path>>(file_path : P) -> io::Result<DataMatrix<T>> {
        let mut data: Vec<T> = Vec::new();
        let mut rows = 0;
        let mut cols = 0;
        let file = try!(File::open(file_path));
        let reader = io::BufReader::new(&file);
        let mut cell_count;
        let mut line;
        for line_read in reader.lines() {
            line = try!(line_read);
            rows += 1;
            cell_count = 0usize;
            for cell in line.split_whitespace() {
                cell_count += 1;
                let cell_val = match T::from_str(cell) {
                    Ok(val) => val,
                    Err(_) => 
                        return Err(
                            io::Error::new(
                                io::ErrorKind::Other, 
                                format!("Could not parse cell with value {}", cell)
                            )
                        )
                };
                data.push(cell_val);
            }
            if rows == 1 {
                cols = cell_count;
            } else {
                if cell_count != cols {
                    return Err(io::Error::new(io::ErrorKind::Other, "Matrix size is not uniform"));
                }
            }
        }
        Ok(DataMatrix{ data: data, shape: (rows, cols) })
    }

    fn print(&self) {
        let (rows, cols) = self.shape;
        for i in 0..rows {
            let mut line = String::new();
            for j in 0..cols {
                if j > 0 { line.push(' '); }
                let cell = self.data[i * cols + j].to_string();
                line = line + &cell;
            }
            println!("{}", line);
        }
    }

    fn get(&self, row: usize, col: usize) -> &T {
        let (_, cols) = self.shape;
        &(self.data[row * cols + col])
    }

    fn set(&mut self, row: usize, col: usize, val: T) {
        let (_, cols) = self.shape;
        self.data[row * cols + col] = val;
    }

    fn as_dmat(&self) -> DMat<T> {
        let (rows, cols) = self.shape;
        let mut out = DMat::new_zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                out[(i,j)] = self.data[i*cols + j];
            }
        }
        out
    }
}


struct MultinomialNB {
    // alpha: f32,
    // fit_prior: bool,
    // shape: n_classes
    class_log_prior: DVec<f64>,
    class_count: DVec<usize>,
    // shape: (n_classes, n_features) 
    feature_log_prob: DMat<f64>,
    feature_count: DMat<usize>,
}


impl MultinomialNB {
    fn fit(x: DMat<usize>, y: DMat<usize>) -> Result<MultinomialNB, &'static str> {
        let n_records = x.nrows();
        let n_feats = x.ncols();
        let n_classes = y.ncols();
        if y.nrows() != n_records {
            return Err("Data matrices have mismatched record count");
        }
        let mut clf = MultinomialNB {
            // alpha: alpha,
            // fit_prior: fit_prior,
            class_log_prior: DVec::new_zeros(n_classes),
            class_count: DVec::new_zeros(n_classes),
            feature_log_prob: DMat::new_zeros(n_classes, n_feats),
            feature_count: DMat::new_zeros(n_classes, n_feats),
        };
        // Accumulate Counts
        let mut n_observations = 0usize;
        for i in 0..n_records {
            for j in 0..n_classes {
                if y[(i,j)] == 1 {
                    for k in 0..n_feats {
                        let n = x[(i,k)];
                        n_observations += n;
                        clf.class_count[j] += n;
                        clf.feature_count[(j,k)] += n;
                    }
                } else if y[(i,j)] != 0 {
                    return Err("Non-binary value found in y");
                }
            }
        }
        // compute log probabilities
        let log_n_observations = (n_observations as f64).log2();
        for c in 0..n_classes {
            // pr(c) = class_count[c] / n_observations
            clf.class_log_prior[c] = (clf.class_count[c] as f64).log2() - log_n_observations;
            for f in 0..n_feats {
                // pr(f | c) = feature_count[c,f] / class_count[c]
                clf.feature_log_prob[(c,f)] = 
                    (clf.feature_count[(c,f)] as f64).log2() - (clf.class_count[c] as f64).log2();
            }
        }
        Ok(clf)
    }

    fn predict(&self, x: DMat<usize>) -> Result<DMat<f64>, &'static str> {
        let n_records = x.nrows();
        let n_features = x.ncols();
        let n_classes = self.class_count.len();
        if n_features != self.feature_count.ncols() {
            return Err("Data Matrix has mismatched feature count");
        }
        // pr(C|F) = pr(F|C) * pr(C) / pr(F)
        // pr(C|F) = product(pr(F_i|C), i) * pr(C) / pr(F)
        let mut y = DMat::new_zeros(n_records, n_classes);
        for i in 0..n_records {
            let mut divisor = 0f64;
            for j in 0..n_classes {
                let mut log_dividend = self.class_log_prior[j];
                for k in 0..n_features {
                    log_dividend += self.feature_log_prob[(j,k)];
                }
                y[(i, j)] = log_dividend.exp2();
                divisor += log_dividend.exp2();
                
            }
            // sum(pr(C|F), C) = 1
            for j in 0..n_classes {
                y[(i,j)] /= divisor;
            }
        }
        Ok(y)
    }
}


fn main() {
    let argv: Vec<String> = env::args().collect();
    let x_file_path = argv[1].clone();
    let y_file_path = argv[2].clone();
    let z_file_path = argv[3].clone();

    let x: DMat<usize> = DataMatrix::open(&x_file_path).unwrap().as_dmat();
    let y: DMat<usize> = DataMatrix::open(&y_file_path).unwrap().as_dmat();
    let z: DMat<usize> = DataMatrix::open(&z_file_path).unwrap().as_dmat();

    let clf = MultinomialNB::fit(x, y).unwrap();
    println!("{:?}", clf.predict(z).unwrap());
}
