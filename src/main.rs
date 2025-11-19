use image::{DynamicImage, GenericImageView, ImageFormat, imageops};
use rand::Rng;
use std::fs::File;
use std::io;
use std::io::{BufWriter, Write};
use std::path::Path;

fn main() {
    let mut choice = String::new();
    println!("0 for train, 1 for test: ");

    io::stdin()
        .read_line(&mut choice)
        .expect("Failed to read line");

    let choice: u32 = choice.trim().parse().expect("Please type a number!");
    if choice == 0 {
        train();
    } else {
        test();
    }
}

fn compute(inp: &[f32], theta: &[f32]) -> i32 {
    let mut sum = 0.0;
    for i in 0..inp.len() {
        sum += inp[i] * theta[i];
    }
    if sum <= 0.0 { -1 } else { 1 }
}

fn perceptron(inp: &[f32], theta: &mut Vec<f32>, label: i32) {
    let result = compute(inp, theta);
    if result != label {
        for i in 0..inp.len() {
            theta[i] += (label as f32) * inp[i];
        }
    }
}

fn train() {
    let mut label: i32;
    let mut inp: Vec<f32>;
    let theta: &mut Vec<f32> = &mut vec![0.0; 10001];
    generate_random_theta(theta); // 32*32*3 + 1 bias
    for i in 0..1000 {
        (inp, label) = get_train_image_vector(i);
        perceptron(&inp, theta, label);
    }
}

fn get_train_image_vector(i: i32) -> (Vec<f32>, i32) {
    let path: String;
    let label: i32;
    if i % 2 == 0 {
        path = format!("dataset/cat.{}.png", 3500 + i / 2);
        label = 1; // cat is +1
    } else {
        path = format!("dataset/dog.{}.png", 3500 + (i - 1) / 2);
        label = -1; // dog is -1 
    }
    let img = image::open(&Path::new(&path)).unwrap();
    let mut inp: Vec<f32> = img.as_bytes().iter().map(|&p| p as f32).collect();
    normalize(&mut inp);
    (inp, label)
}

fn get_test_image_vector(i: i32) -> (Vec<f32>, i32) {
    let path: String;
    let label: i32;
    if i % 2 == 0 {
        path = format!("dataset/cat.{}.png", i / 2);
        label = 1; // cat is +1
    } else {
        path = format!("dataset/dog.{}.png", (i - 1) / 2);
        label = -1; // dog is -1 
    }
    let img = image::open(&Path::new(&path)).unwrap();
    let mut inp: Vec<f32> = img.as_bytes().iter().map(|&p| p as f32).collect();
    normalize(&mut inp);
    (inp, label)
}

fn normalize(data: &mut Vec<f32>) {
    for i in 0..data.len() {
        data[i] = 2.0 * data[i] / 255.0 - 1.0;
    }
    data.push(1.0); // bias term
}

fn generate_random_theta(theta: &mut Vec<f32>) {
    let mut rng = rand::rng();

    let range = rand::distr::Uniform::new(-0.01, 0.01);

    theta.clear();
    // theta.extend((0..10001).map(|_| range.sample(&rng)));
}

fn test() {
    let mut index = String::new();
    let label: i32;
    let inp: Vec<f32>;
    let theta: &mut Vec<f32> = &mut vec![0.0; 10001];
    println!("enter index: ");

    io::stdin()
        .read_line(&mut index)
        .expect("Failed to read line");

    let index: i32 = index.trim().parse().expect("Please type a number!");
    (inp, label) = get_test_image_vector(index);
    read_theta(theta);
    let prediction = compute(&inp, theta);
    println!("Prediction: {}, Actual: {}", prediction, label);
    if label == 1 {
        println!("Actual: Cat")
    } else {
        println!("Actual: Dog")
    }
}

fn read_theta(theta: &mut Vec<f32>) {}

fn write_theta(theta: &Vec<f32>) {
    let file = File::create("theta.txt");
    let mut writer = BufWriter::new(file);

    // Write each value on a new line
    for value in theta {
        writeln!(writer, "{}", value);
    }
}
