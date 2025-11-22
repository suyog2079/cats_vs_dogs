use image::{DynamicImage, GenericImageView, ImageFormat, imageops};
use rand::Rng;
use std::fs::File;
use std::io::{self, Read};
use std::io::{BufWriter, Write};
use std::path::Path;

fn main() {
    let mut choice = String::new();
    println!("0 for train, 1 for test, 2 for evaluation: ");

    io::stdin()
        .read_line(&mut choice)
        .expect("Failed to read line");

    let choice: u32 = choice.trim().parse().expect("Please type a number!");
    if choice == 0 {
        train();
    } else if choice == 1 {
        test();
    } else {
        evaluate();
    }

    // let (img, label) = get_train_image_vector(100);
    // println!("Image size: {}", img.len());
}

fn evaluate() {
    let theta: &mut Vec<f32> = &mut vec![0.0; 80601];
    read_theta(theta);
    let mut correct = 0;
    for i in 0..100 {
        let (inp, label) = get_test_image_vector(i);
        let prediction = compute(&inp, theta);
        if prediction == label {
            correct += 1;
        }
    }
    println!("Accuracy: {}%", correct as f32 / 10.0);
}

fn compute(inp: &[f32], theta: &mut Vec<f32>) -> i32 {
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
    let mut theta = vec![0.0; 80601];
    generate_random_theta(&mut theta);
    for i in 0..1000 {
        let (inp, label) = get_train_image_vector(i);
        perceptron(&inp, &mut theta, label);
        println!("{}\n", i);
    }
    write_theta(&theta);
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
    let img = image::open(&Path::new(&path)).unwrap().to_luma8();
    let img = image::imageops::resize(&img, 20, 20, imageops::FilterType::Gaussian);
    let mut inp: Vec<f32> = img.as_raw().iter().map(|&p| p as f32).collect();
    let original_len = inp.len();
    for i in 0..original_len {
        for j in i..original_len {
            inp.push(inp[i] * inp[j]);
        }
    }
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
    let img = image::open(&Path::new(&path)).unwrap().to_luma8();
    let img = image::imageops::resize(&img, 20, 20, imageops::FilterType::Gaussian);
    let mut inp: Vec<f32> = img.as_raw().iter().map(|&p| p as f32).collect();
    let original_len = inp.len();
    for i in 0..original_len {
        for j in i..original_len {
            inp.push(inp[i] * inp[j]);
        }
    }
    normalize(&mut inp);
    (inp, label)
}

fn normalize(data: &mut Vec<f32>) {
    for i in 0..data.len() {
        data[i] = 2.0 * data[i] / 255.0 - 1.0;
    }
    data.push(1.0); // bias term
    // println!("length: {}", data.len());
}

fn generate_random_theta(theta: &mut Vec<f32>) {
    let mut rng = rand::rng();

    let range = rand::distr::Uniform::new(-0.01, 0.01).unwrap();
    *theta = (&mut rng)
        .sample_iter(range)
        .take(80601)
        .collect::<Vec<f32>>();
}

fn test() {
    let mut index = String::new();
    let label: i32;
    let inp: Vec<f32>;
    let theta: &mut Vec<f32> = &mut vec![0.0; 80601];
    println!("enter index: ");

    io::stdin()
        .read_line(&mut index)
        .expect("Failed to read line");

    let index: i32 = index.trim().parse().expect("Please type a number!");
    (inp, label) = get_test_image_vector(index);
    read_theta(theta);
    let prediction = compute(&inp, theta);
    if label == 1 {
        println!("Actual: Cat");
    } else {
        println!("Actual: Dog")
    }
    if prediction == 1 {
        println!("Prediction: Cat");
    } else {
        println!("Prediction: Dog");
    }
}

fn read_theta(theta: &mut Vec<f32>) {
    let buf = std::fs::read("theta.bin").expect("Cannot read theta.bin");

    *theta = buf
        .chunks_exact(4)
        .map(|chunk| {
            let arr: [u8; 4] = chunk.try_into().unwrap();
            f32::from_le_bytes(arr)
        })
        .collect();
}

fn write_theta(theta: &Vec<f32>) {
    let mut file = File::create("theta.bin").expect("Cannot create theta.bin");

    for &val in theta {
        let bytes = val.to_le_bytes(); // f32 → [u8; 4]
        file.write_all(&bytes).unwrap();
    }
}
