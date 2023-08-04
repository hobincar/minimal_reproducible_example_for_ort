use std::path::Path;
use std::sync::Arc;
use std::vec::Vec;

use ndarray::{CowArray, Dim, IxDynImpl, arr1};
use ort::{
    tensor::OrtOwnedTensor,
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder, Value
};


fn func() -> OrtResult<OrtOwnedTensor<'static, f32, Dim<IxDynImpl>>> {
    let environment = Arc::new(
        Environment::builder()
            .with_execution_providers([ExecutionProvider::CPU(Default::default())])
            .build()?
    );
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_model_from_file(Path::new(&String::from("softmax.onnx")))?;

    let input = CowArray::from(arr1(&[1f32, 2f32, 3f32, 4f32])).into_dyn();

    let output: Vec<Value> = session.run(vec![
        Value::from_array(session.allocator(), &input)?,
    ])?;
    let output: OrtOwnedTensor<f32, _> = output[0].try_extract()?;
    println!("[1] output: {:?}", output);

    let output = Ok(output);
    println!("[2] output: {:?}", output);

    output
}


fn main() {
    let output = func();

    println!("[3] output: {:?}", output);
}

