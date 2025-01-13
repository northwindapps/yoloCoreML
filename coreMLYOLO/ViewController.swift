import CoreML
import Vision
import UIKit
import AVFoundation

class ViewController: UIViewController, AVCaptureVideoDataOutputSampleBufferDelegate {
    var captureSession: AVCaptureSession!
    var previewLayer: AVCaptureVideoPreviewLayer!
    var model: VNCoreMLModel?
    var imageView: UIImageView!
    var shapeLayers: [CAShapeLayer] = []
    var textLayers: [CATextLayer] = []
    var lastPredictionTime: Date = Date()
    var capturedImages:[UIImage] = []
    var counter = 0
    var maxLimit = 5

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadModel()
        setupCamera()
    }

    func setupUI() {
        imageView = UIImageView(frame: view.bounds)
        imageView.contentMode = .scaleAspectFit
        imageView.layer.zPosition = 1
        view.addSubview(imageView)
    }

    func loadModel() {
        do {
            guard let modelURL = Bundle.main.url(forResource: "best", withExtension: "mlmodelc") else {
                fatalError("Failed to find the model file.")
            }
            let coreMLModel = try MLModel(contentsOf: modelURL)
            self.model = try VNCoreMLModel(for: coreMLModel)
        } catch {
            fatalError("Failed to load model: \(error.localizedDescription)")
        }
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .photo

        guard let camera = AVCaptureDevice.default(for: .video) else { return }
        let cameraInput = try? AVCaptureDeviceInput(device: camera)

        if captureSession.canAddInput(cameraInput!) {
            captureSession.addInput(cameraInput!)
        }

        let videoOutput = AVCaptureVideoDataOutput()
        videoOutput.setSampleBufferDelegate(self, queue: DispatchQueue(label: "videoQueue"))
        captureSession.addOutput(videoOutput)

        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.isHidden = false
        view.layer.addSublayer(previewLayer)

        captureSession.startRunning()
    }

    func predict(image: UIImage) {
        guard let ciImage = CIImage(image: image), let model = model else {
            fatalError("Unable to create CIImage from UIImage or model not loaded")
        }
        
        
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        let request = VNCoreMLRequest(model: model) { request, error in
            if let error = error {
                print("Failed to perform request: \(error.localizedDescription)")
                return
            }
            self.processResults(request.results, in: image)
        }

        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform request: \(error.localizedDescription)")
        }
    }


    func processResults(_ results: [Any]?, in image: UIImage) {
        guard let results = results as? [VNRecognizedObjectObservation] else {
            print("No results or results are of unexpected type")
            return
        }
        
        if counter > maxLimit{
            print("Reached the limit")
            return
        }

        // Fix the orientation of the image
        let fixedImage = image
        let imageSize = fixedImage.size

        UIGraphicsBeginImageContextWithOptions(imageSize, false, 0.0)
        fixedImage.draw(in: CGRect(origin: .zero, size: imageSize))

        //save image
        //capturedImages.append(fixedImage)
        for observation in results {
            let boundingBox = observation.boundingBox
            let rect = CGRect(
                x: boundingBox.origin.x * imageSize.width,
                y: (1 - boundingBox.origin.y - boundingBox.height) * imageSize.height,
                width: boundingBox.width * imageSize.width,
                height: boundingBox.height * imageSize.height
            )

            UIColor.red.setStroke()
            UIRectFrame(rect)
            
            //
            // Convert UIImage to CGImage
            guard let cgImage = fixedImage.cgImage else { return }
            
            // Crop the image using the rect
            guard let croppedCGImage = cgImage.cropping(to: rect) else { return }
            
            // Convert cropped CGImage back to UIImage
            let croppedImage = UIImage(cgImage: croppedCGImage)
            //capturedImages.append(croppedImage)
            
            
            if let label = observation.labels.first?.identifier {
                let image = croppedImage
                performOCR(on: image) { recognizedText in
                    if let text = recognizedText {
                        print("Key:\(label),Value: \(text)")
                        self.counter += 1
                    } else {
                        print("No text recognized")
                    }
                }
            }

            
        }
        
    }

    func performOCR(on image: UIImage, completion: @escaping ([String]?) -> Void) {
        // Convert UIImage to CGImage
        guard let cgImage = image.cgImage else {
            completion(nil)
            return
        }

        // Create a request for text recognition
        let request = VNRecognizeTextRequest { (request, error) in
            if let error = error {
                print("Error during OCR: \(error)")
                completion(nil)
                return
            }

            // Extract recognized text
            let recognizedStrings = request.results?.compactMap { result -> String? in
                guard let observation = result as? VNRecognizedTextObservation else { return nil }
                return observation.topCandidates(1).first?.string
            }

            completion(recognizedStrings)
        }

        // Set recognition level and language (optional)
        request.recognitionLevel = .accurate // .accurate or .fast
        request.recognitionLanguages = ["en-US"] // You can add more languages

        // Create a request handler
        let requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])

        // Perform the request
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try requestHandler.perform([request])
            } catch {
                print("Error performing OCR request: \(error)")
                completion(nil)
            }
        }
    }



    func convertCIImageToUIImage(ciImage: CIImage) -> UIImage? {
        let context = CIContext(options: nil)
        if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
            return UIImage(cgImage: cgImage)
        }
        return nil
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        connection.videoOrientation = AVCaptureVideoOrientation.portrait
        let currentTime = Date()
        if currentTime.timeIntervalSince(lastPredictionTime) < 1.0 {
            return
        }
        lastPredictionTime = currentTime
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let uiImage = convertCIImageToUIImage(ciImage: ciImage) else {
            return
        }
        
//        Run prediction in background thread to avoid lag
        DispatchQueue.global(qos: .userInitiated).async {
            self.predict(image: uiImage)
        }
    }
}

