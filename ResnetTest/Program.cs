using ResnetTest;


if (args.Length == 0)
{
    Console.WriteLine("Please provide the path to an image file as a command-line argument.");
    return;
}

var imagePath = args[0];

var modelPath = "resnet50.onnx";
var labelsPath = "labels.txt";

var classifier = new ImageClassifier(modelPath, labelsPath);
var result = classifier.Predict(imagePath);

Console.WriteLine($"This is most likely: {result}");