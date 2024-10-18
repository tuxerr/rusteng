// compile shaders
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("Compiling shaders with slang");
    let asset_dir = Path::new("./shaders");
    assert!(env::set_current_dir(&asset_dir).is_ok());
    println!("Current path : {}", env::current_dir().unwrap().into_os_string().into_string().unwrap());
    println!("Current execution path : {}",env::var("PATH").unwrap());

    fs::read_dir(".").unwrap()
        .filter(| file | file.as_ref().unwrap().file_name().to_str().unwrap().ends_with(".slang"))
        .for_each(|shader| {
            let shader_name = shader.unwrap().file_name().into_string().unwrap();
            let subfields: Vec<&str> = shader_name.split(".").collect();
            let shader_name_noext = subfields.first().unwrap();

            println!("Compiling {} into VS and FS", &shader_name);

            Command::new("slangc")
                .args(&[shader_name.as_str(), "-profile", "sm_6_0", "-entry", "vertexMain", "-target", "spirv", "-o"])
                .arg(&format!("vs{}.o", shader_name_noext)).spawn().expect("slang VS failure").wait().expect("Failure to wait");

            Command::new("slangc")
                .args(&[shader_name.as_str(), "-profile", "sm_6_5", "-entry", "meshMain", "-target", "spirv", "-o"])
                .arg(&format!("mesh{}.o", shader_name_noext)).spawn().expect("slang VS failure").wait().expect("Failure to wait");

            Command::new("slangc")
                .args(&[shader_name.as_str(), "-profile", "sm_6_0", "-entry", "fragmentMain", "-target", "spirv", "-o"])
                .arg(&format!("fs{}.o", shader_name_noext))
                .spawn().expect("slang FS failure").wait().expect("Failure to wait");

            Command::new("slangc")
                .args(&[shader_name.as_str(), "-profile", "sm_6_0", "-entry", "computeMain", "-target", "spirv", "-o"])
                .arg(&format!("cs{}.o", shader_name_noext))
                .spawn().expect("slang CS failure").wait().expect("Failure to wait");
        });
}