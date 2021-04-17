/* 
*   Classifier Cam
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.Examples {

    using UnityEngine;
    using UnityEngine.UI;
    using System.Threading.Tasks;
    using NatSuite.Devices;
    using NatSuite.ML;
    using NatSuite.ML.Features;
    using NatSuite.ML.Vision;

    public class ClassifierCam : MonoBehaviour {

        public enum Classifier { MobileNet, ShuffleNet, SqueezeNet }

        [Header("Classification")]
        public Classifier classifier;
        
        [Header(@"UI")]
        public RawImage rawImage;
        public AspectRatioFitter aspectFitter;
        public Text labelText;
        public Text confidenceText;

        private ICameraDevice cameraDevice;
        private MLClassifier model;
        private MLDispatcher<(string, float)> dispatcher;
        private Texture2D preview;
        
        async void Start () {
            // Request camera permissions
            if (!await MediaDeviceQuery.RequestPermissions<ICameraDevice>()) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Create a classifier
            var modelPath = await ModelPath(classifier);
            var classLabels = await MLClassifier.LoadLabelsFromStreamingAssets(@"classes.txt");
            model = new MLClassifier(modelPath, classLabels);
            // Create a dispatcher to run predictions on a worker thread
            dispatcher = MLModelUtility.CreateDispatcher(model.Classify);
            // Get the default camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.GenericCameraDevice);
            cameraDevice = query.current as ICameraDevice;
            // Start the camera preview
            cameraDevice.previewResolution = (1280, 720);
            preview = await cameraDevice.StartRunning();
            // Display
            rawImage.texture = preview;
            aspectFitter.aspectRatio = (float)preview.width / preview.height;
        }

        void Update () {
            // Check that the camera is running
            if (!preview)
                return;
            // Check if dispatcher can accept new work
            if (!dispatcher.readyForPrediction)
                return;
            // Create image feature for classification
            var imageFeature = new MLImageFeature(preview);
            imageFeature.mean = new Vector3(0.485f, 0.456f, 0.406f);
            imageFeature.std = new Vector3(0.229f, 0.224f, 0.225f);
            // Classify
            dispatcher.Predict(imageFeature, OnClassification);
        }

        void OnClassification ((string, float) result) {
            // Display
            var (label, confidence) = result;
            labelText.text = label?.Split(' ')[1].Split(',')[0].Trim();
            confidenceText.text = $"{confidence:#.##}";
        }

        void OnDestroy () {
            // Stop camera
            if (cameraDevice?.running ?? false)
                cameraDevice.StopRunning();
            // Dispose dispatcher and model
            if (model != null) {
                dispatcher.Dispose();
                model.Dispose();
            }
        }

        static async Task<string> ModelPath (Classifier classifier) {
            // Get relative path in `StreamingAssets`
            var relativePath = "";
            switch (classifier) {
                case Classifier.MobileNet: relativePath = @"mobilenetv2-7.onnx"; break;
                case Classifier.ShuffleNet: relativePath = @"shufflenet-v2-10.onnx"; break;
                case Classifier.SqueezeNet: relativePath = @"squeezenet1.1-7.onnx"; break;
                default: relativePath = @"mobilenetv2-7.onnx"; break;
            }
            // Get full path on file system
            var modelPath = await MLModelUtility.ModelPathFromStreamingAssets(relativePath);
            return modelPath;
        }
    }
}