use std::io::prelude::*;
use std::io;
use std::fs::File;
use std::env;

fn load_data(file_path : String) -> io::Result<String> {
    let mut file = try!(File::open(file_path));
    let mut s = String::new();
    try!(file.read_to_string(&mut s));
    Ok(s)
}

fn main() {
    let argv: Vec<String> = env::args().collect();
    let X_file_path = argv[1].clone();
    let y_file_path = argv[2].clone();
    println!("X: {}", X_file_path);
    println!("Y: {}", y_file_path);
}
