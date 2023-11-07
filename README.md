# siamese networks for duplicate or near-duplicate image detection

**Dataset**: https://www.kaggle.com/datasets/barelydedicated/airbnb-duplicate-image-detection

Siamese Neural Network (SNN) is a neural network architecture containing two or more identical subnetworks. “Identical” here means, they have the same configuration with the same parameters and weights. The update of parameters is reflected on both its subnets simultaneously.


SNN is used to find the similarity of input data by comparing their feature vectors. Some popular applications of SNN include: Face Verification, Signature Verification, Image Seaching System, ...

![image](https://github.com/nttrung2406/siamese-networks-for-duplicate-near-duplicate-image-detection/assets/105348335/e5719fb5-10b5-4edf-bacb-66ab9221f9e0)

**The working flow of SNN**

Select a pair of Input Data (in the scope of this article, images) selected from the dataset.

Pass each image through each Sub-network of SNN for processing. The output of the Sub-networks is an Embedding vector.

Calculate the Euclidean distance between those 2 Embedding vectors.

A Sigmoid Function can be applied on the distance to give a Score value in the interval [0,1], representing the degree of similarity between two Embedding vectors. The closer the score is to 1, the more similar the two vectors are and vice versa.

**Loss function**

a) Triple Loss function

The idea of Triple Loss is to use a set of 3 Input Data including: Anchor (A), Positive (P) and Negative (N) in which the distance from A to P is minimized, while the distance from A is minimized. to N is maximized during model training.

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>L</mi>
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mo>,</mo>
  <mi>P</mi>
  <mo>,</mo>
  <mi>N</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <mi>m</mi>
  <mi>a</mi>
  <mi>x</mi>
  <mo stretchy="false">(</mo>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>f</mi>
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2212;</mo>
  <mi>f</mi>
  <mo stretchy="false">(</mo>
  <mi>P</mi>
  <mo stretchy="false">)</mo>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>2</mn>
  </msup>
  <mo>&#x2212;</mo>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <mi>f</mi>
  <mo stretchy="false">(</mo>
  <mi>A</mi>
  <mo stretchy="false">)</mo>
  <mo>&#x2212;</mo>
  <mi>f</mi>
  <mo stretchy="false">(</mo>
  <mi>N</mi>
  <mo stretchy="false">)</mo>
  <mrow data-mjx-texclass="ORD">
    <mo stretchy="false">|</mo>
  </mrow>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mo stretchy="false">|</mo>
    </mrow>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <mi>&#x3B1;</mi>
  <mo>,</mo>
  <mn>0</mn>
  <mo stretchy="false">)</mo>
</math>

b) Contrastive loss

The idea of Contrastive Loss is similar to Triplet Loss, the difference is that Contrastive Loss only uses 1 pair of Input Data, either the same type, or different types. If they are of the same type, the distance between their feature vectors will be minimized, and if they are different types, the distance between their feature vectors will be maximized during the training process.

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mo stretchy="false">(</mo>
  <mn>1</mn>
  <mo>&#x2212;</mo>
  <mi>Y</mi>
  <mo stretchy="false">)</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <mo stretchy="false">(</mo>
  <msub>
    <mi>D</mi>
    <mi>w</mi>
  </msub>
  <msup>
    <mo stretchy="false">)</mo>
    <mn>2</mn>
  </msup>
  <mo>+</mo>
  <mo stretchy="false">(</mo>
  <mi>Y</mi>
  <mo stretchy="false">)</mo>
  <mfrac>
    <mn>1</mn>
    <mn>2</mn>
  </mfrac>
  <msup>
    <mrow data-mjx-texclass="ORD">
      <mi>m</mi>
      <mi>a</mi>
      <mi>x</mi>
      <mo stretchy="false">(</mo>
      <mn>0</mn>
      <mo>,</mo>
      <mi>m</mi>
      <mo>&#x2212;</mo>
      <msub>
        <mi>D</mi>
        <mi>w</mi>
      </msub>
      <mo stretchy="false">)</mo>
    </mrow>
    <mn>2</mn>
  </msup>
</math>

D_w is Euclidean distance

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <msqrt>
    <msup>
      <mrow data-mjx-texclass="ORD">
        <msub>
          <mi>G</mi>
          <mi>w</mi>
        </msub>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>X</mi>
          <mn>1</mn>
        </msub>
        <mo stretchy="false">)</mo>
        <mo>&#x2212;</mo>
        <msub>
          <mi>G</mi>
          <mi>w</mi>
        </msub>
        <mo stretchy="false">(</mo>
        <msub>
          <mi>X</mi>
          <mn>2</mn>
        </msub>
        <mo stretchy="false">)</mo>
      </mrow>
      <mn>2</mn>
    </msup>
  </msqrt>
</math>



