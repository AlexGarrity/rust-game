use std::{io::Write, path::PathBuf};

use fs_extra::copy_items;
use fs_extra::dir::CopyOptions;
use glob::glob;
use shaderc::{Compiler, ShaderKind};

const ASSETS_DIR: &'static str = "res";
const BUILD_DIR_ENV_NAME: &'static str = "OUT_DIR";

fn compile_shader_file(
    compiler: &Compiler,
    path: &PathBuf,
    shader_kind: ShaderKind,
) -> Result<(), &'static str> {
    // TODO - Make this actually compile shaders
    let file_path = path.to_str().unwrap();
    let file_contents = std::fs::read(file_path).expect("Failed to read shader source file");
    let source =
        String::from_utf8(file_contents).expect("Shader source file contains invalid characters");
    let compilation_result = compiler
        .compile_into_spirv(source.as_str(), shader_kind, file_path, "main", None)
        .expect("Shader file could not be compiled");
    let spirv = compilation_result.as_binary_u8();

    let mut file_writer = std::fs::OpenOptions::new()
        .truncate(true)
        .write(true)
        .open(path.join(".spv"))
        .expect("Failed to open writer for SPIR-V file");
    file_writer
        .write_all(spirv)
        .expect("Failed to write spirv file");

    Ok(())
}

fn compile_shader_files() -> Result<(), &'static str> {
    let compiler = Compiler::new().expect("Failed to create shader compiler");

    for vertex_src in glob("client/res/shaders/*.vert").unwrap() {
        match vertex_src {
            Ok(path) => {
                if compile_shader_file(&compiler, &path, ShaderKind::Vertex).is_err() {
                    return Err("Failed to compile a vertex shader");
                }
            }
            Err(_error) => {
                return Err("A glob match was invalid");
            }
        }
    }

    for fragment_src in glob("client/res/shaders/*.frag").unwrap() {
        match fragment_src {
            Ok(path) => {
                if compile_shader_file(&compiler, &path, ShaderKind::Fragment).is_err() {
                    return Err("Failed to compile a fragment shader");
                }
            }
            Err(_error) => {
                return Err("A glob match was invalid");
            }
        }
    }

    Ok(())
}

fn main() -> Result<(), String> {
    // println!("cargo:rerun-if-changed={}/*", ASSETS_DIR);

    let compilation_result = compile_shader_files();
    if compilation_result.is_err() {
        return Err(format!("Failed to compile all shaders"));
    }

    let out_dir = std::env::var(BUILD_DIR_ENV_NAME).unwrap();
    let build_dir = format!("{}/../../..", out_dir);
    let copy_options = CopyOptions::new().overwrite(true);
    let copy_result = copy_items(&[ASSETS_DIR], &build_dir, &copy_options);

    match copy_result {
        Ok(_count) => Ok(()),
        Err(error) => Err(format!(
            "Failed to copy files to {} (error code {})",
            build_dir, error
        )),
    }
}
