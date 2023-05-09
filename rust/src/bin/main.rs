use tract_onnx::prelude::*;

fn main() {
    let path = std::env::args().nth(1).expect("no path to a model.pt given");
    println!("loading model: ${path:?}");
    let model = tract_onnx::onnx()
        .model_for_path(path).unwrap()
        .into_optimized().unwrap()
        .into_runnable().unwrap();
    let qty_model_inputs = model.model().inputs.len();
    println!("Number of inputs needed by this model: {:?}", qty_model_inputs);
    // Define the input tensor shape
    let model_inputs = &model.model().inputs;
    for (input_idx, &outlet_id ) in model_inputs.iter().enumerate() {
        let node_id = outlet_id.node;
        let input_name = &model.model().nodes.get(node_id).unwrap().name;
        let input_shape = &model.model().input_fact(input_idx).unwrap().shape.as_concrete().unwrap().clone();
        println!("input {} {:?} shape {:?}", input_idx, input_name, input_shape);
    }

    let length = 50;

    let mut zero_vec: Vec<i64> = vec![0; length];
    let session_items: Vec<i64> = vec![1, 5, 7, 3];

    for (i, &value) in session_items.iter().enumerate() {
        if i < zero_vec.len() {
            zero_vec[i] = value;
        } else {
            break;
        }
    }
    let input_ids: Tensor = tract_ndarray::Array2::from_shape_vec(
        (1, length),
        zero_vec,
    ).unwrap()
        .into();

    let item_mask: Tensor = Tensor::from(tract_ndarray::arr1(&[session_items.len() as i64]));

    let model_input = if qty_model_inputs == 0 {
        tvec!()
    } else if qty_model_inputs == 1 {
        tvec!(input_ids.into())
    } else {
        // tvec!(input_ids.into())
        tvec!(input_ids.into(), item_mask.into())
    };

    let outputs =
        model.run(model_input).unwrap();
    // let outputs =
    //     model.run(tvec!(input_ids.into(), item_mask.into())).unwrap();

    println!("outputs: {outputs:?}");
}