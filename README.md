# minimal_reproducible_example_for_ort

## How to Run
```bash
$ ORT_DYLIB_PATH=<YOUR_ORT_DYLIB_PATH> cargo build
$ cargo run
```

## Output
```
    Finished dev [unoptimized + debuginfo] target(s) in 0.05s
    Finished dev [unoptimized + debuginfo] target(s) in 0.05s
     Running `target/debug/ort_minimal_reproducible_example`
[1] output: OrtOwnedTensor { data: TensorPtr { ptr: 0xaaab0b67e7d0, array_view: [0.032058604, 0.08714432, 0.23688284, 0.6439143], shape=[4], strides=[1], layout=CFcf (0xf), dynamic ndim=1 } }
[2] output: Ok(OrtOwnedTensor { data: TensorPtr { ptr: 0xaaab0b67e7d0, array_view: [0.032058604, 0.08714432, 0.23688284, 0.6439143], shape=[4], strides=[1], layout=CFcf (0xf), dynamic ndim=1 } })
[3] output: Ok(OrtOwnedTensor { data: TensorPtr { ptr: 0xaaab0b67e7d0, array_view: [0.0, 0.0, 3.124826e-32, 6.1224e-41], shape=[4], strides=[1], layout=CFcf (0xf), dynamic ndim=1 } })
```

## softmax.onnx
from [netron](https://netron.app/)

![softmax onnx](https://github.com/hobincar/minimal_reproducible_example_for_ort/assets/17702664/00cd34cd-428b-4ed0-9a89-605757894e70)
