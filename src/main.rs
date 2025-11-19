use image::{DynamicImage, GenericImageView, ImageFormat, imageops};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_images = 4000;

    for i in 450..500 {
        let input_name = format!("Asirra/dog.{}.jpg",i);

        // Check file exists (optional but helpful)
        if !Path::new(&input_name).exists() {
            println!("Skipping missing file: {}", input_name);
            continue;
        }

        // ---- Load image ----
        let img = image::open(&input_name)?;
        
        // ---- Resize to 50×50 ----
        let resized = img.resize_exact(100, 100, imageops::FilterType::Lanczos3);

        // ---- Convert to grayscale ----
        let gray = resized.to_luma8();

        // ---- Save output ----
        let output_name = format!("dataset/dog.{}.png", i+3500);
        gray.save(&output_name)?;

        println!("Processed {}", output_name);
    }

    println!("Done!");

    Ok(())
}

