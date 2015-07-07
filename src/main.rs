use std::io::prelude::*;
use std::io;
use std::fs::File;
use std::path::Path;
use std::str::FromStr;
use std::string::ToString;
use std::env;

struct DataMatrix<T> {
    data: Vec<T>,
    shape: (usize, usize),
}

impl<T: FromStr + ToString> DataMatrix<T> {
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
}

fn main() {
    let argv: Vec<String> = env::args().collect();
    let x_file_path = argv[1].clone();
    let y_file_path = argv[2].clone();

    let x: DataMatrix<f32> = DataMatrix::open(&x_file_path).unwrap();
    let y: DataMatrix<f32> = DataMatrix::open(&y_file_path).unwrap();

    println!("X:");
    x.print();
    println!("\nY:");
    y.print();
}
