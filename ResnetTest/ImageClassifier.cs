using System;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace ResnetTest
{
    internal class ImageClassifier
    {
        private readonly InferenceSession _session;
        private readonly string[] _classLabels;

        public ImageClassifier(string modelPath, string labelsPath)
        {
            //We want to use DirectML on the main GPU
            var options = new SessionOptions();
            options.AppendExecutionProvider_DML(0);

            //Create the session, passing along the model and session options
            _session = new InferenceSession(modelPath, options);
            
            // Load class labels from a file (each line contains a label)
            _classLabels = File.ReadAllLines(labelsPath);
        }

        /// <summary>
        /// Use the model to give a label to the image
        /// </summary>
        /// <param name="imagePath">Path to the image file</param>
        /// <returns>A label string</returns>
        public string Predict(string imagePath)
        {
            var inputTensor = PreprocessImage(imagePath);

            // Run inference
            var inputs = new List<NamedOnnxValue>
                {
                    NamedOnnxValue.CreateFromTensor("image_tensor", inputTensor) // use the correct input name here
                };
            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);

            // Get the output and convert it to an array of probabilities
            var output = results.First().AsTensor<float>().ToArray();

            // Find the index of the maximum probability
            int maxIndex = output.ToList().IndexOf(output.Max());

            // Map the index to a label
            return _classLabels[maxIndex];
        }

        /// <summary>
        /// Prepare an image for analysis by resizing it and convert it to a tensor
        /// </summary>
        /// <param name="imagePath"></param>
        /// <returns>A tensor that represents the image</returns>
        private DenseTensor<float> PreprocessImage(string imagePath)
        {
            const int targetWidth = 224;
            const int targetHeight = 224;
            var tensor = new DenseTensor<float>(new[] { 1, 3, targetHeight, targetWidth });

            using var image = Image.Load<Rgb24>(imagePath);
            image.Mutate(x => x.Resize(targetWidth, targetHeight));

            for (int y = 0; y < targetHeight; y++)
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    var pixel = image[x, y];
                    tensor[0, 0, y, x] = pixel.R / 255.0f;
                    tensor[0, 1, y, x] = pixel.G / 255.0f;
                    tensor[0, 2, y, x] = pixel.B / 255.0f;
                }
            }

            return tensor;
        }
    }
}
