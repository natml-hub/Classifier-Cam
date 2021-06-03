/* 
*   Classifier Cam
*   Copyright (c) 2021 Yusuf Olokoba.
*/

namespace NatSuite.Examples {

    using UnityEngine;
    using UnityEngine.UI;
    using NatSuite.Devices;
    using NatSuite.ML;
    using NatSuite.ML.Features;
    using NatSuite.ML.Vision;

    public class ClassifierCam : MonoBehaviour {
        
        [Header(@"UI")]
        public RawImage rawImage;
        public AspectRatioFitter aspectFitter;
        public Text labelText;
        public Text confidenceText;

        CameraDevice cameraDevice;
        Texture2D previewTexture;
        MLModelData modelData;
        MLModel model;
        MLClassificationPredictor predictor;
        
        async void Start () {
            // Request camera permissions
            if (!await MediaDeviceQuery.RequestPermissions<CameraDevice>()) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Get the default camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            cameraDevice.previewResolution = (1280, 720);
            previewTexture = await cameraDevice.StartRunning();
            // Display the camera preview
            rawImage.texture = previewTexture;
            aspectFitter.aspectRatio = (float)previewTexture.width / previewTexture.height;
            // Fetch MobileNet classifier
            Debug.Log("Fetching model from NatML Hub");
            modelData = await MLModelData.FromHub("@natsuite/mobilenet-v2");
            model = modelData.Deserialize();
            predictor = new MLClassificationPredictor(model, modelData.labels);
        }

        void Update () {
            // Check that model has been downloaded
            if (predictor == null)
                return;
            // Classify
            var input = new MLImageFeature(previewTexture);
            (input.mean, input.std) = modelData.normalization;
            var (label, confidence) = predictor.Predict(input);
            // Display
            labelText.text = label;
            confidenceText.text = $"{confidence:#.##}";
        }

        void OnDestroy () {
            // Stop camera
            if (cameraDevice?.running ?? false)
                cameraDevice.StopRunning();
            // Dispose model
            model?.Dispose();
        }
    }
}