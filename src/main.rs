use image::imageops;
use rand::Rng;
use std::fs::File;
use std::io::Write;
use std::io::{self};
use std::path::Path;

// const IMG_SIZE: usize = 20 * 20; // 400
// const POLY_SIZE: usize = IMG_SIZE * (IMG_SIZE + 1) / 2; // 80400
// const TOTAL_FEATURES: usize = 401; // 80401

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
    let (inp, _label) = get_test_image_vector(0);
    let theta = &mut vec![0.0; inp.len()];
    read_theta(theta);
    let mut correct = 0;
    for i in 0..100 {
        let (inp, label) = get_test_image_vector(i);
        let prediction = if compute(&inp, theta) >= 0.5 { 1 } else { 0 };
        if prediction == label {
            correct += 1;
        }
    }
    println!("Accuracy: {}%", correct as f32);
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn compute(inp: &[f32], theta: &[f32]) -> f32 {
    let sum: f32 = inp.iter().zip(theta.iter()).map(|(x, t)| x * t).sum();

    sigmoid(sum)
}

fn train() {
    let (inp, _label) = get_train_image_vector(0);
    let theta = &mut vec![0.0; inp.len()];
    let mut guess: f32;
    let mut sum: Vec<f32> = vec![0.0; inp.len()];

    let epochs = 100;
    let lambda = 0.5;
    let learning_rate = 0.05;
    generate_random_theta(theta);
    for j in 0..epochs {
        println!("{}", j + 1);
        for i in 0..1000 {
            let (inp, label) = get_train_image_vector(i);
            guess = compute(&inp, theta);
            sum = sum
                .iter()
                .zip(inp.iter())
                .map(|(s, &x)| s + x * (guess - label as f32))
                .collect();
        }
        let eta = learning_rate;
        for k in 0..theta.len() {
            theta[k] -= eta * sum[k] / 1000.0 - eta * lambda * theta[k];
        }
    }
    write_theta(&theta);
}

fn get_train_image_vector(i: i32) -> (Vec<f32>, i32) {
    let path: String;
    let label: i32;
    if i % 2 == 0 {
        path = format!("dataset/cat.{}.png", 3500 + i / 2);
        label = 0; // cat is 0 
    } else {
        path = format!("dataset/dog.{}.png", 3500 + (i - 1) / 2);
        label = 1; // dog is 1 
    }
    let img = image::open(&Path::new(&path)).unwrap().to_luma8();
    let img = image::imageops::resize(&img, 20, 20, imageops::FilterType::Gaussian);
    let mut inp: Vec<f32> = img.as_raw().iter().map(|&p| p as f32).collect();
    let original_len = inp.len();
    normalize(&mut inp);
    for i in 0..original_len {
        for j in i..original_len {
            inp.push(inp[i] * inp[j]);
        }
    }
    inp.push(1.0); // bias term
    (inp, label)
}

fn get_test_image_vector(i: i32) -> (Vec<f32>, i32) {
    let path: String;
    let label: i32;
    if i % 2 == 0 {
        path = format!("dataset/cat.{}.png", i / 2);
        label = 0; // cat is 0 
    } else {
        path = format!("dataset/dog.{}.png", (i - 1) / 2);
        label = 1; // dog is 1 
    }
    let img = image::open(&Path::new(&path)).unwrap().to_luma8();
    let img = image::imageops::resize(&img, 20, 20, imageops::FilterType::Gaussian);
    let mut inp: Vec<f32> = img.as_raw().iter().map(|&p| p as f32).collect();
    let original_len = inp.len();
    normalize(&mut inp);
    for i in 0..original_len {
        for j in i..original_len {
            inp.push(inp[i] * inp[j]);
        }
    }
    inp.push(1.0); // bias term
    (inp, label)
}

fn normalize(data: &mut Vec<f32>) {
    for i in 0..data.len() {
        data[i] = 2.0 * data[i] / 255.0 - 1.0;
    }
}

fn generate_random_theta(theta: &mut Vec<f32>) {
    let mut rng = rand::rng();
    let (inp, _label) = get_train_image_vector(0);

    let range = rand::distr::Uniform::new(-0.01, 0.01).unwrap();
    *theta = (&mut rng)
        .sample_iter(range)
        .take(inp.len())
        .collect::<Vec<f32>>();
}

fn test() {
    let mut index = String::new();
    let label: i32;
    let inp: Vec<f32>;
    println!("enter index: ");

    io::stdin()
        .read_line(&mut index)
        .expect("Failed to read line");

    let index: i32 = index.trim().parse().expect("Please type a number!");
    (inp, label) = get_test_image_vector(index);
    let theta = &mut vec![0.0; inp.len()];
    read_theta(theta);
    let prediction = compute(&inp, theta);
    if label == 0 {
        println!("Actual: Cat");
    } else {
        println!("Actual: Dog")
    }
    if prediction < 0.5 {
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
