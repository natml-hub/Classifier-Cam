/* 
*   Classifier Cam
*   Copyright (c) 2022 NatML Inc. All Rights Reserved.
*/

namespace NatSuite.Examples {

    using UnityEngine;
    using UnityEngine.UI;
    using NatSuite.Devices;
    using NatSuite.Devices.Outputs;
    using NatSuite.ML;
    using NatSuite.ML.Features;
    using NatSuite.ML.Vision;

    public class ClassifierCam : MonoBehaviour {
            
        [Header(@"NatML")]
        public string accessKey;

        [Header(@"UI")]
        public RawImage rawImage;
        public AspectRatioFitter aspectFitter;
        public Text labelText;
        public Text confidenceText;

        CameraDevice cameraDevice;
        TextureOutput textureOutput;
        MLModelData modelData;
        MLModel model;
        MobileNetv2Predictor predictor;
        
        async void Start () {
            // Request camera permissions
            var permissionStatus = await MediaDeviceQuery.RequestPermissions<CameraDevice>();
            if (permissionStatus != PermissionStatus.Authorized) {
                Debug.LogError(@"User did not grant camera permissions");
                return;
            }
            // Get the default camera device
            var query = new MediaDeviceQuery(MediaDeviceCriteria.CameraDevice);
            cameraDevice = query.current as CameraDevice;
            // Start the camera preview
            textureOutput = new TextureOutput();
            cameraDevice.previewResolution = (1280, 720);
            cameraDevice.StartRunning(textureOutput);
            // Display the camera preview
            var previewTexture = await textureOutput;
            rawImage.texture = previewTexture;
            aspectFitter.aspectRatio = (float)previewTexture.width / previewTexture.height;
            // Fetch the model data from NatML Hub
            Debug.Log("Fetching model from NatML Hub");
            modelData = await MLModelData.FromHub("@natsuite/mobilenet-v2", accessKey);
            // Deserialize the model
            model = modelData.Deserialize();
            // Create the MobileNet v2 predictor
            predictor = new MobileNetv2Predictor(model, modelData.labels);
        }

        void Update () {
            // Check that model has been downloaded
            if (predictor == null)
                return;
            // Create input feature
            var previewTexture = textureOutput.texture;
            var inputFeature = new MLImageFeature(previewTexture.GetRawTextureData<byte>(), previewTexture.width, previewTexture.height);
            (inputFeature.mean, inputFeature.std) = modelData.normalization;
            // Classify
            var (label, confidence) = predictor.Predict(inputFeature);
            // Display
            labelText.text = label;
            confidenceText.text = $"{confidence:#.##}";
        }

        void OnDestroy () {
            // Stop camera
            textureOutput?.Dispose();
            if (cameraDevice?.running ?? false)
                cameraDevice.StopRunning();
            // Dispose model
            model?.Dispose();
        }
    }
}