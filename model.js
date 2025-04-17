const tf = require("@tensorflow/tfjs");
const model = tf.sequential();
model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });


 async function train(xs, ys, epochs = 10) {
    const xTensor = tf.tensor(xs);
    const yTensor = tf.tensor(ys);
    await model.fit(xTensor, yTensor, { epochs });
  }
  function predict(input) {
    const createTensor = tf.tensor1d([input])
   const result =  model.predict(createTensor);
   return result.dataSync()[0];
  }
  async function fitandpredict(xs , ys , input) {
  await  train(xs , ys , epochs = 100);
     const output = predict(input);
     console.log(output);
     
      

  }
  const xs = [1 ,2 ,3 ,4];
  const ys = [1 ,3 , 5 , 7];
  fitandpredict(xs , ys , 5);
  
