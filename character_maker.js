async function initialization(){
    console.log("waiting for session initalization...")
    // Create an ONNX inference session with default backend.
    const session = new onnx.InferenceSession({backendHint : "webgl"});
    await session.loadModel("./export_model_light.onnx");

    console.log("success initialization")
    let dummy_x = new Float32Array(1* 3 * 224 * 224).fill(0);
    let dummy_inp = new onnx.Tensor(dummy_x, 'float32', [1,3, 224, 224]);
    let dummy = await session.run([dummy_inp]);

    window.session = session
}

async function faceInference(input_init) {
    let inp = new onnx.Tensor(input_init, 'float32', [1,3, 224, 224]);

    // Run model with Tensor inputs and get the result by output name defined in model.
    const outputMap = await window.session.run([inp]);
    return outputMap.values().next().value.data;
}