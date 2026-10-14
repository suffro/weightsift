//! Builds the C test client of the store's ABI (`tests/c/abi_client.c`, compiled against `include/weightsift_store.h`
//! by the platform's C compiler) and links it into the integration tests only: it never enters the library's artifacts.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=include/weightsift_store.h");
    println!("cargo:rerun-if-changed=tests/c/abi_client.c");
    let out = std::env::var("OUT_DIR").expect("cargo sets OUT_DIR");
    cc::Build::new()
        .file("tests/c/abi_client.c")
        .include("include")
        .warnings(true)
        .extra_warnings(true)
        .warnings_into_errors(true)
        .cargo_metadata(false)
        .compile("ws_abi_client");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").is_ok_and(|env| env == "msvc");
    let archive = if msvc {
        "ws_abi_client.lib"
    } else {
        "libws_abi_client.a"
    };
    println!("cargo:rustc-link-arg-tests={out}/{archive}");
}
