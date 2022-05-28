# **Transformer Applications**

## **Learning Objectives**

* How to use Transformers for Machine Translation
* How to write a custom train/test loop in Keras
* How to use Tensorflow Datasets


## **TF Datasets**

For `machine translation`, we will be using the prepared `Tensorflow Datasets` `ted_hrlr_translate/pt_to_en` for English to Portuguese translation

* ted_hrlr_translate/pt_to_en: from Tensorflow Dataset catalog, [link](https://www.tensorflow.org/datasets/catalog/overview#translate).

## **Masks in Transformers**

* The mask has different shapes depending on its type(padding or look ahead)
* An appropriate mask must be used in the attention step.

## **Useful Resources**

[Transformer model for language understanding - Tensorflow](https://www.tensorflow.org/text/tutorials/transformer#masking)